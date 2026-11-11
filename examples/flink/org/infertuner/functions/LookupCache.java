package org.infertuner.functions;

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.TreeMap;


public class LookupCache {
    private static final class Entry {
        String value;
        int count;

        Entry(String value, int count) {
            this.value = value;
            this.count = count;
        }
    }

    private final Map<String, Entry> map = new HashMap<>();
    private final TreeMap<Integer, LinkedHashSet<String>> byCount = new TreeMap<>();
    private long capacityBytes;
    private long usedBytes;

    public LookupCache(long capacityBytes) {
        this.capacityBytes = capacityBytes;
    }

    public static long sizeOf(String key, String value) {
        return 2L * (key.length() + value.length());
    }

    public String get(String key) {
        Entry e = map.get(key);
        if (e == null) {
            return null;
        }
        touch(key, e);
        return e.value;
    }

    public void put(String key, String value) {
        Entry e = map.get(key);
        if (e != null) {
            usedBytes += sizeOf(key, value) - sizeOf(key, e.value);
            e.value = value;
            touch(key, e);
        } else {
            map.put(key, new Entry(value, 1));
            byCount.computeIfAbsent(1, c -> new LinkedHashSet<>()).add(key);
            usedBytes += sizeOf(key, value);
        }
        evict();
    }

    
    public void resetCounts() {
        LinkedHashSet<String> all = new LinkedHashSet<>();
        for (LinkedHashSet<String> keys : byCount.values()) {
            all.addAll(keys);
        }
        byCount.clear();
        for (Entry e : map.values()) {
            e.count = 0;
        }
        if (!all.isEmpty()) {
            byCount.put(0, all);
        }
    }

    public void resize(long newCapacityBytes) {
        capacityBytes = newCapacityBytes;
        evict();
    }

    public long capacityBytes() {
        return capacityBytes;
    }

    public long usedBytes() {
        return usedBytes;
    }

    public int size() {
        return map.size();
    }

    private void touch(String key, Entry e) {
        LinkedHashSet<String> keys = byCount.get(e.count);
        keys.remove(key);
        if (keys.isEmpty()) {
            byCount.remove(e.count);
        }
        e.count++;
        byCount.computeIfAbsent(e.count, c -> new LinkedHashSet<>()).add(key);
    }

    private void evict() {
        while (usedBytes > capacityBytes && !byCount.isEmpty()) {
            Map.Entry<Integer, LinkedHashSet<String>> lowest = byCount.firstEntry();
            Iterator<String> it = lowest.getValue().iterator();
            String key = it.next();
            it.remove();
            if (lowest.getValue().isEmpty()) {
                byCount.remove(lowest.getKey());
            }
            Entry e = map.remove(key);
            usedBytes -= sizeOf(key, e.value);
        }
    }
}
