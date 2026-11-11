package org.infertuner.functions;

import java.util.Arrays;
import java.util.Collection;


public final class CacheCapacityEstimator {
    private CacheCapacityEstimator() {}

    public static double targetHitRate(double tHit, double tMiss, double lambdaInstance) {
        if (lambdaInstance <= 0 || tMiss <= tHit) {
            return 0.0;
        }
        double h = (tMiss - 1.0 / lambdaInstance) / (tMiss - tHit);
        return Math.min(1.0, Math.max(0.0, h));
    }

    
    public static int requiredKeys(Collection<Integer> counts, double hTarget) {
        if (counts.isEmpty() || hTarget <= 0) {
            return 0;
        }
        int[] f = counts.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(f);
        long total = 0;
        for (int v : f) {
            total += v;
        }
        long acc = 0;
        int d = 0;
        for (int i = f.length - 1; i >= 0; i--) {
            acc += f[i];
            d++;
            if ((double) acc / total >= hTarget - 1e-12) {
                break;
            }
        }
        return d;
    }

    
    public static long newCapacity(Collection<Integer> counts, double tHit, double tMiss, double lambdaInstance,
                                   double avgEntryBytes, long currentBytes, double epsilon,
                                   long minBytes, long maxBytes) {
        int d = requiredKeys(counts, targetHitRate(tHit, tMiss, lambdaInstance));
        long s = Math.max(minBytes, Math.min(maxBytes, (long) Math.ceil(d * avgEntryBytes)));
        if (currentBytes > 0 && Math.abs(s - currentBytes) / (double) currentBytes <= epsilon) {
            return -1;
        }
        return s;
    }
}
