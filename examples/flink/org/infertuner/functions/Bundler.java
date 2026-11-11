package org.infertuner.functions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.functions.co.BroadcastProcessFunction;
import org.apache.flink.util.Collector;
import org.infertuner.control.Reporter;
import org.infertuner.model.Bundle;
import org.infertuner.model.Event;


public class Bundler extends BroadcastProcessFunction<Event, Integer, Bundle> implements CheckpointedFunction {
    public static final MapStateDescriptor<String, Integer> CONTROL =
            new MapStateDescriptor<>("control", Types.STRING, Types.INT);

    private final Reporter reporter;
    private final int initialB;

    private transient List<Event> buffer;
    private transient int b;
    private transient int nextB;
    private transient int subtask;
    private transient long seq;
    private transient ListState<Event> bufferState;

    public Bundler(Reporter reporter, int initialB) {
        this.reporter = reporter;
        this.initialB = initialB;
        if (initialB < 1) throw new IllegalArgumentException("bundle size must be positive");
    }

    @Override
    public void open(Configuration parameters) {
        subtask = getRuntimeContext().getIndexOfThisSubtask();
        reporter.startReporting("events");
    }

    @Override
    public void close() {
        reporter.close();
    }

    @Override
    public void processElement(Event e, ReadOnlyContext ctx, Collector<Bundle> out) throws Exception {
        Integer current = ctx.getBroadcastState(CONTROL).get("bundle_size");
        if (current != null) {
            nextB = current;
        }
        if (buffer.isEmpty()) b = nextB;
        e.joinMs = System.currentTimeMillis();
        Map<String, Object> r = new HashMap<>();
        r.put("id", e.id);
        r.put("log_ms", e.logMs);
        r.put("arrival_ms", e.joinMs);
        r.put("pre_ms", e.joinMs - e.promptMs);
        r.put("prompt_len", e.promptTokens);
        r.put("pred_len", e.predTokens);
        reporter.report(r);
        buffer.add(e);
        emitFull(out);
    }

    @Override
    public void processBroadcastElement(Integer newB, Context ctx, Collector<Bundle> out) throws Exception {
        if (newB < 1) throw new IllegalArgumentException("bundle size must be positive");
        ctx.getBroadcastState(CONTROL).put("bundle_size", newB);
        nextB = newB;
        if (buffer.isEmpty()) b = nextB;
        emitFull(out);
    }

    private void emitFull(Collector<Bundle> out) {
        while (buffer.size() >= b) {
            List<Event> events = new ArrayList<>(buffer.subList(0, b));
            buffer.subList(0, b).clear();
            out.collect(new Bundle(subtask, seq++, events));
            b = nextB;
        }
    }

    @Override
    public void snapshotState(FunctionSnapshotContext context) throws Exception {
        bufferState.update(buffer);
    }

    @Override
    public void initializeState(FunctionInitializationContext context) throws Exception {
        bufferState = context.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>("bundle-buffer", Event.class));
        buffer = new ArrayList<>();
        b = initialB;
        nextB = initialB;
        seq = 0;
        if (context.isRestored()) {
            for (Event e : bufferState.get()) {
                buffer.add(e);
            }
        }
    }
}
