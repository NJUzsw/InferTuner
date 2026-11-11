package org.infertuner.functions;

import java.io.Serializable;


public interface DimensionTable extends Serializable {
    void open() throws Exception;

    String read(String key) throws Exception;

    


    class Simulated implements DimensionTable {
        private final long latencyMs;
        private final int valueChars;

        public Simulated(long latencyMs, int valueChars) {
            this.latencyMs = latencyMs;
            this.valueChars = valueChars;
        }

        @Override
        public void open() {}

        @Override
        public String read(String key) throws InterruptedException {
            Thread.sleep(latencyMs);
            StringBuilder sb = new StringBuilder("content of ").append(key).append(": ");
            while (sb.length() < valueChars) {
                sb.append(key);
            }
            return sb.substring(0, valueChars);
        }
    }
}
