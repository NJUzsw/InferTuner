package org.infertuner.model;

import java.util.ArrayList;
import java.util.List;


public class Bundle {
    public int subtask;
    public long seq;
    public List<Event> events = new ArrayList<>();

    public Bundle() {}

    public Bundle(int subtask, long seq, List<Event> events) {
        this.subtask = subtask;
        this.seq = seq;
        this.events = events;
    }
}
