package org.infertuner.control;

import java.io.Serializable;
import java.util.Map;


public class Reporter implements Serializable, AutoCloseable {
    
    public void startReporting(String kind) {}

    public void report(Map<String, Object> item) {}

    @Override
    public void close() {}
}
