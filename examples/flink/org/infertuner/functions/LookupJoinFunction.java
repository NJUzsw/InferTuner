package org.infertuner.functions;

import java.util.HashMap;
import java.util.Map;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.metrics.Gauge;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.infertuner.control.Reporter;
import org.infertuner.model.Event;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class LookupJoinFunction extends RichMapFunction<Event, Event> implements CheckpointedFunction {
    private static final Logger LOG = LoggerFactory.getLogger(LookupJoinFunction.class);

    private final DimensionTable table;
    private final long initialBytes, minBytes, maxBytes;
    private final long windowMs;
    private final double epsilon;
    private final boolean adaptive;
    private final Reporter reporter;

    private transient LookupCache cache;
    private transient long restoredBytes;
    private transient Map<String, Integer> counts;
    private transient long windowStartMs, arrivals, hits, misses, hitNanos, missNanos;
    private transient double tHit, tMiss;           // seconds, kept from earlier windows if a window has no samples
    private transient double entryBytesSum;
    private transient long entries;
    private transient volatile double lastHitRate;
    private transient ListState<Long> capacityState;

    
    public LookupJoinFunction(DimensionTable table, long initialBytes, long minBytes, long maxBytes,
                              long windowMs, double epsilon, boolean adaptive, Reporter reporter) {
        this.table = table;
        this.initialBytes = initialBytes;
        this.minBytes = minBytes;
        this.maxBytes = maxBytes;
        this.windowMs = windowMs;
        this.epsilon = epsilon;
        this.adaptive = adaptive;
        this.reporter = reporter;
    }

    @Override
    public void open(Configuration parameters) throws Exception {
        table.open();
        cache = new LookupCache(restoredBytes > 0 ? restoredBytes : initialBytes);
        counts = new HashMap<>();
        windowStartMs = System.currentTimeMillis();
        tHit = 1e-6;
        tMiss = 1e-2;
        reporter.startReporting("cache");
        getRuntimeContext().getMetricGroup().gauge("lookupCacheBytes", (Gauge<Long>) () -> cache.capacityBytes());
        getRuntimeContext().getMetricGroup().gauge("lookupCacheHitRate", (Gauge<Double>) () -> lastHitRate);
    }

    @Override
    public void close() {
        reporter.close();
    }

    @Override
    public Event map(Event e) throws Exception {
        long now = System.currentTimeMillis();
        if (now - windowStartMs >= windowMs) {
            endOfWindow(now);
        }
        long t0 = System.nanoTime();
        String v = cache.get(e.key);
        if (v != null) {
            hits++;
            hitNanos += System.nanoTime() - t0;
        } else {
            v = table.read(e.key);
            cache.put(e.key, v);
            misses++;
            missNanos += System.nanoTime() - t0;
            entryBytesSum += LookupCache.sizeOf(e.key, v);
            entries++;
        }
        counts.merge(e.key, 1, Integer::sum);
        arrivals++;
        e.lookedUp = v;
        return e;
    }

    private void endOfWindow(long now) {
        double seconds = (now - windowStartMs) / 1000.0;
        double lambda = arrivals / seconds;
        if (hits > 0) {
            tHit = hitNanos / 1e9 / hits;
        }
        if (misses > 0) {
            tMiss = missNanos / 1e9 / misses;
        }
        lastHitRate = arrivals > 0 ? (double) hits / arrivals : 0.0;
        double avgEntry = entries > 0 ? entryBytesSum / entries : 0.0;
        double hTarget = CacheCapacityEstimator.targetHitRate(tHit, tMiss, lambda);
        if (adaptive && arrivals > 0 && avgEntry > 0) {
            long s = CacheCapacityEstimator.newCapacity(counts.values(), tHit, tMiss, lambda, avgEntry,
                    cache.capacityBytes(), epsilon, minBytes, maxBytes);
            if (s >= 0) {
                LOG.info("lookup cache {}: lambda'={}/s t_h={}s t_m={}s hit={} h_target={} resize {} -> {} bytes",
                        getRuntimeContext().getIndexOfThisSubtask(), lambda, tHit, tMiss, lastHitRate, hTarget,
                        cache.capacityBytes(), s);
                cache.resize(s);
            }
        }
        Map<String, Object> r = new HashMap<>();
        r.put("time_ms", now);
        r.put("subtask", getRuntimeContext().getIndexOfThisSubtask());
        r.put("capacity_bytes", cache.capacityBytes());
        r.put("hit_rate", lastHitRate);
        r.put("lambda", lambda);
        r.put("h_target", hTarget);
        r.put("t_hit", tHit);
        r.put("t_miss", tMiss);
        reporter.report(r);
        counts.clear();
        cache.resetCounts();
        arrivals = hits = misses = hitNanos = missNanos = 0;
        windowStartMs = now;
    }

    @Override
    public void snapshotState(FunctionSnapshotContext context) throws Exception {
        capacityState.clear();
        if (cache != null) {
            capacityState.add(cache.capacityBytes());
        }
    }

    @Override
    public void initializeState(FunctionInitializationContext context) throws Exception {
        capacityState = context.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>("cache-capacity", Long.class));
        restoredBytes = 0;
        if (context.isRestored()) {
            for (Long v : capacityState.get()) {
                restoredBytes = Math.max(restoredBytes, v);
            }
        }
    }
}
