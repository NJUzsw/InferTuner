package org.infertuner.engine;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;


public class EngineClient {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final int ATTEMPTS = 3;

    private final String baseUrl;
    private final HttpClient http;

    public EngineClient(String baseUrl) {
        this.baseUrl = baseUrl.replaceAll("/+$", "");
        this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
    }

    public String url() {
        return baseUrl;
    }

    public CompletableFuture<JsonNode> post(String path, Object body) {
        String json;
        try {
            json = JSON.writeValueAsString(body);
        } catch (IOException e) {
            return CompletableFuture.failedFuture(e);
        }
        return attempt(path, json, 1);
    }

    private CompletableFuture<JsonNode> attempt(String path, String json, int n) {
        HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + path))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();
        return http.sendAsync(req, HttpResponse.BodyHandlers.ofString())
                .thenApply(r -> {
                    if (r.statusCode() != 200) {
                        throw new IllegalStateException("engine " + baseUrl + path + " returned " + r.statusCode() + ": " + r.body());
                    }
                    try {
                        return JSON.readTree(r.body());
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                })
                .handle((v, err) -> {
                    if (err == null) {
                        return CompletableFuture.completedFuture(v);
                    }
                    if (n >= ATTEMPTS) {
                        return CompletableFuture.<JsonNode>failedFuture(err);
                    }
                    return CompletableFuture.supplyAsync(() -> json,
                                    CompletableFuture.delayedExecutor(1000L * n, TimeUnit.MILLISECONDS))
                            .thenCompose(j -> attempt(path, j, n + 1));
                })
                .thenCompose(f -> f);
    }
}
