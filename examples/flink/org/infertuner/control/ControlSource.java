package org.infertuner.control;

import java.io.Serializable;

import org.apache.flink.streaming.api.functions.source.RichSourceFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class ControlSource extends RichSourceFunction<Integer> {
    private static final Logger LOG = LoggerFactory.getLogger(ControlSource.class);

    
    public interface BundleSize extends Serializable {
        int latest() throws Exception;
    }

    private final BundleSize bundleSize;
    private final long pollMs;
    private volatile boolean running = true;

    public ControlSource(BundleSize bundleSize, long pollMs) {
        this.bundleSize = bundleSize;
        this.pollMs = pollMs;
    }

    @Override
    public void run(SourceContext<Integer> ctx) throws Exception {
        int last = -1;
        while (running) {
            try {
                int b = bundleSize.latest();
                if (b != last) {
                    last = b;
                    synchronized (ctx.getCheckpointLock()) {
                        ctx.collect(b);
                    }
                }
            } catch (Exception e) {
                LOG.warn("cannot read the bundle size: {}", e.toString());
            }
            Thread.sleep(pollMs);
        }
    }

    @Override
    public void cancel() {
        running = false;
    }
}
