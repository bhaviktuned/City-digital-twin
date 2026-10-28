const { spawn } = require("child_process");
const readline = require("readline");
const { WebSocketServer } = require("ws");

const city = process.argv[2] || "noida";
const py = spawn(process.env.PYTHON || "python", ["sim.py", "--city", city], { stdio: ["pipe", "pipe", "inherit"] });
const wss = new WebSocketServer({ port: 8080 });
const allowed = new Set(["density", "workers", "toggle", "reopen", "tps", "pause", "reset", "closeArea"]);
let netMsg = null;

readline.createInterface({ input: py.stdout }).on("line", line => {
  if (line.startsWith('{"type":"net"')) netMsg = line;
  for (const c of wss.clients) if (c.readyState === 1) c.send(line);
});

wss.on("connection", ws => {
  if (netMsg) ws.send(netMsg);
  ws.on("message", raw => {
    try {
      const m = JSON.parse(raw);
      if (allowed.has(m.cmd)) py.stdin.write(JSON.stringify(m) + "\n");
    } catch {}
  });
});

py.on("exit", () => process.exit(0));
process.on("SIGINT", () => { py.kill(); process.exit(0); });
console.log("relay on ws://localhost:8080 (" + city + ")");
