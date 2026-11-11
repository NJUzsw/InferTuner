package org.infertuner.functions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeoutException;

import com.fasterxml.jackson.databind.JsonNode;
import org.apache.flink.api.common.functions.RuntimeContext;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.async.ResultFuture;
import org.apache.flink.streaming.api.functions.async.RichAsyncFunction;
import org.infertuner.control.Reporter;
import org.infertuner.engine.EngineClient;
import org.infertuner.model.Bundle;
import org.infertuner.model.Event;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class InferenceFunction extends RichAsyncFunction<Bundle, Bundle> {
    private static final Logger LOG = LoggerFactory.getLogger(InferenceFunction.class);

    private final String[] engineUrls;
    private final Reporter reporter;
    private transient EngineClient engine;

    public InferenceFunction(String[] engineUrls, Reporter reporter) {
        this.engineUrls = engineUrls;
        this.reporter = reporter;
    }

    static String engineOf(RuntimeContext ctx, String[] urls) {
        if (ctx.getNumberOfParallelSubtasks() > urls.length) {
            throw new IllegalStateException("inference parallelism " + ctx.getNumberOfParallelSubtasks()
                    + " exceeds the number of engines " + urls.length);
        }
        return urls[ctx.getIndexOfThisSubtask()];
    }

    @Override
    public void open(Configuration parameters) {
        engine = new EngineClient(engineOf(getRuntimeContext(), engineUrls));
        reporter.startReporting("bundles");
        LOG.info("inference instance {} uses {}", getRuntimeContext().getIndexOfThisSubtask(), engine.url());
    }

    @Override
    public void close() {
        reporter.close();
    }

    @Override
    public void asyncInvoke(Bundle bundle, ResultFuture<Bundle> result) {
        List<String> prompts = new ArrayList<>();
        for (Event e : bundle.events) {
            prompts.add(e.prompt);
        }
        engine.post("/generate", Map.of("prompts", prompts)).whenComplete((resp, err) -> {
            if (err != null) {
                result.completeExceptionally(err);
                return;
            }
            long end = System.currentTimeMillis();
            JsonNode texts = resp.get("texts");
            JsonNode tokens = resp.get("output_tokens");
            List<Long> ids = new ArrayList<>();
            List<Integer> outLens = new ArrayList<>();
            for (int i = 0; i < bundle.events.size(); i++) {
                Event e = bundle.events.get(i);
                e.output = texts.get(i).asText();
                e.outputTokens = tokens.get(i).asInt();
                e.inferEndMs = end;
                ids.add(e.id);
                outLens.add(e.outputTokens);
            }
            Map<String, Object> r = new HashMap<>();
            r.put("subtask", bundle.subtask);
            r.put("seq", bundle.seq);
            r.put("ids", ids);
            r.put("output_tokens", outLens);
            r.put("prompt_tokens", resp.get("prompt_tokens").asInt());
            r.put("engine_ms", resp.get("engine_ms").asDouble());
            r.put("end_ms", end);
            reporter.report(r);
            result.complete(Collections.singletonList(bundle));
        });
    }

    @Override
    public void timeout(Bundle bundle, ResultFuture<Bundle> result) {
        result.completeExceptionally(new TimeoutException("bundle " + bundle.subtask + "/" + bundle.seq + " timed out"));
    }
}
