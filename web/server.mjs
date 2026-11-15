import http from "node:http";
import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("../", import.meta.url));
let active = false;
function python(args, input) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      process.env.PYTHON || "python3",
      ["src/simulation.py", ...args],
      { cwd: root, stdio: ["pipe", "pipe", "pipe"] },
    );
    let output = "",
      errors = "";
    const timeout = setTimeout(() => {
      child.kill();
      reject(new Error("Simulation timed out."));
    }, 120000);
    child.stdout.on("data", (data) => {
      output += data;
    });
    child.stderr.on("data", (data) => {
      errors += data;
    });
    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(new Error(`Python 3.10+ is required: ${error.message}`));
    });
    child.on("close", (code) => {
      clearTimeout(timeout);
      try {
        const value = JSON.parse(output);
        if (code !== 0)
          reject(new Error(value.error || errors || "Simulation failed"));
        else if (errors.includes("WARNING") || errors.includes("TimeoutError"))
          reject(new Error(errors));
        else resolve(value);
      } catch {
        reject(new Error(errors || "Invalid simulation response"));
      }
    });
    child.stdin.on("error", () => {});
    child.stdin.end(JSON.stringify(input || {}));
  });
}
const files = {
  "/": ["index.html", "text/html"],
  "/app.js": ["app.js", "text/javascript"],
  "/style.css": ["style.css", "text/css"],
};
const server = http.createServer(async (req, res) => {
  const send = (status, data) => {
    res.writeHead(status, { "Content-Type": "application/json" });
    res.end(JSON.stringify(data));
  };
  try {
    const url = new URL(req.url, "http://localhost");
    if (url.pathname === "/api/bots" && req.method === "GET")
      return send(200, await python(["--catalog"]));
    if (url.pathname === "/api/simulate" && req.method === "POST") {
      if (active)
        return send(429, {
          error: "A game is being prepared. Try again in a moment.",
        });
      let body = "";
      for await (const chunk of req) {
        body += chunk;
        if (body.length > 8192)
          return send(413, { error: "Request too large" });
      }
      let config;
      try {
        config = JSON.parse(body);
      } catch {
        return send(400, { error: "Invalid JSON" });
      }
      if (!config || typeof config !== "object" || Array.isArray(config))
        return send(400, { error: "Invalid configuration" });
      active = true;
      try {
        return send(200, await python([], config));
      } catch (error) {
        return send(400, { error: error.message });
      } finally {
        active = false;
      }
    }
    if (files[url.pathname] && req.method === "GET") {
      const [name, type] = files[url.pathname];
      res.writeHead(200, {
        "Content-Type": `${type}; charset=utf-8`,
        "Cache-Control": "no-cache",
      });
      return res.end(await readFile(path.join(root, "web", name)));
    }
    send(404, { error: "Not found" });
  } catch (error) {
    send(500, { error: error.message });
  }
});
const port = Number(process.env.PORT || 5173);
server.listen(port, "127.0.0.1", () =>
  console.log(`\n  MEGAGEM Observatory\n  Local: http://localhost:${port}\n`),
);
server.on("error", (error) => {
  console.error(error.message);
  process.exitCode = 1;
});
