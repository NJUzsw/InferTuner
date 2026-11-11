package org.infertuner.model;


public class Event {
    public long id;
    
    public String key;
    
    public String text;
    
    public String lookedUp;
    
    public String prompt;
    
    public String output;

    
    public int promptTokens;
    public int predTokens;
    public int outputTokens;

    
    public long logMs;
    
    public long readMs;
    
    public long promptMs;
    
    public long joinMs;
    
    public long inferEndMs;

    public Event() {}

    public Event(long id, String key, String text, long logMs, long readMs) {
        this.id = id;
        this.key = key;
        this.text = text;
        this.logMs = logMs;
        this.readMs = readMs;
    }
}
