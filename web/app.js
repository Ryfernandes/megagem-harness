const $ = (id) => document.getElementById(id);
const colors = ["Blue", "Green", "Purple", "Yellow", "Pink"];
const labels = {
  RandomBidder: "Random · value-bounded",
  AggressiveBidder: "Very aggressive · high bids",
  PassiveBidder: "Very passive · low bids",
  SamplingBot: "Sampling search · level 4",
  SoftTrackingBot: "Soft tracking · informed bids",
  ValuationBot: "Valuation · expected value",
  FixedRuleBot: "Fixed rule · 20%",
  SmallBidBot: "Small bid · 1 coin",
  Bot: "Zero bid · baseline",
};
let catalog = [],
  seats = 4,
  replay = null,
  frame = 0,
  timer = null;
const gem = (g, big = false) =>
  `<i class="gem g${g}${big ? " big" : ""}" role="img" aria-label="${colors[g]} gem" title="${colors[g]}"></i>`;
const escape = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const botLabel = (name) => labels[name] || name;
const level3Features = [
  ["use_bid_evidence", "Hand inference"],
  ["use_endgame_rules", "Mission endgame rules"],
  ["use_finance_horizon", "Finance-aware cash planning"],
  ["use_cash_fraction_model", "Cash-fraction bid model"],
  ["use_strategic_reveal", "Strategic reveals"],
];
function seatFields() {
  const old = [...document.querySelectorAll(".bot-select")].map((s) => s.value);
  const oldOptions = [...document.querySelectorAll(".seat-field")].map((field) =>
    Object.fromEntries([...field.querySelectorAll(".feature-switch")].map((el) => [el.dataset.feature, el.checked])),
  );
  $("seat-fields").innerHTML = Array.from(
    { length: seats },
    (_, i) =>
      `<div class="seat-field"><label for="bot-${i}"><i class="seat-dot s${i}"></i> Seat ${i + 1}</label><select class="bot-select" id="bot-${i}">${catalog.map((name) => `<option value="${escape(name)}" ${name === (old[i] || ["FixedRuleBot", "SmallBidBot", "FixedRuleBot", "Bot", "SmallBidBot"][i]) ? "selected" : ""}>${escape(botLabel(name))}</option>`).join("")}</select><fieldset class="feature-options" hidden><legend>Level 3 extras · off by default</legend>${level3Features.map(([key, label]) => `<label><input type="checkbox" class="feature-switch" data-feature="${key}" ${oldOptions[i]?.[key] ? "checked" : ""}>${label}</label>`).join("")}</fieldset></div>`,
  ).join("");
  document.querySelectorAll(".bot-select").forEach((select) => {
    const update = () => { select.parentElement.querySelector(".feature-options").hidden = select.value !== "SoftTrackingBot"; };
    select.addEventListener("change", update);
    update();
  });
}
function stop() {
  clearInterval(timer);
  timer = null;
  $("play").textContent = "▶ Play";
  if (replay)
    $("status").textContent =
      frame === replay.frames.length - 1
        ? "Game complete"
        : frame
          ? "Paused"
          : "Ready to watch";
}
function play() {
  if (!replay) return;
  if (timer) return stop();
  if (frame === replay.frames.length - 1) frame = 0;
  $("play").textContent = "Ⅱ Pause";
  timer = setInterval(
    () => {
      frame++;
      render();
      if (frame >= replay.frames.length - 1) stop();
    },
    Number($("speed").value),
  );
  render();
}
function title(event) {
  if (!event) return "Table is set";
  if (event.type === "FINAL") return "The final reveal";
  if (event.type === "AUCTION")
    return `${event.gems.length === 2 ? "Two gems" : "One gem"} up for auction`;
  return `${event.type === "LOAN" ? "Borrow" : "Investment return"} ${event.value} coins`;
}
function missionMarkup(m, owner) {
  const patterns = {
    SAME: m.wildCount === 2 ? "Any pair" : "Three of a kind",
    DIFFERENT: `${m.wildCount} different colors`,
    PAIRS: "Any two pairs",
  };
  return `<div class="mission-row ${owner !== undefined ? "claimed" : ""}"><div><div class="mission-gems">${m.gems.map((g) => gem(g)).join("")}${m.wildCount ? '<span class="mission-description">' + "◇ ".repeat(m.wildCount) + "</span>" : ""}</div><span class="mission-description">${m.wildCount ? patterns[m.pattern] || "Any gems" : m.gems.map((g) => colors[g]).join(" + ")}</span>${owner !== undefined ? `<div class="mission-owner">✓ Claimed by Seat ${owner + 1}</div>` : ""}</div><span class="mission-points">+${m.value} <small>pts</small></span></div>`;
}
function render() {
  if (!replay) return;
  const { state: s, event: e } = replay.frames[frame];
  const final = e?.type === "FINAL";
  const total = replay.frames.length - 1;
  $("game-meta").textContent =
    `${s.players.length} seats · Chart ${replay.chart} · Seed ${replay.seed}`;
  $("round-label").textContent = final
    ? "FINAL SCORING"
    : frame
      ? `ROUND ${String(s.round).padStart(2, "0")}`
      : "SETUP";
  $("status").textContent = final
    ? "Game complete"
    : timer
      ? "Playing"
      : "Ready to watch";
  $("remaining").textContent = `${s.deck + s.row.length} gems remaining`;
  $("timeline").max = total;
  $("timeline").value = frame;
  $("frame-label").textContent = `${frame} / ${total}`;
  $("prev").disabled = frame === 0;
  $("next").disabled = frame === total;
  $("play").disabled = false;
  $("timeline").disabled = false;
  $("player-count").textContent =
    `/ ${String(s.players.length).padStart(2, "0")}`;
  let content;
  if (!e) {
    content = `<div class="auction-kicker">THE OPENING LINEUP</div><h3>Let the bidding begin.</h3><div class="lot">${s.row.map((g) => `<div class="gem-card">${gem(g, true)}</div>`).join("")}</div><p>The first two gems are visible. Press play or step forward.</p>`;
  } else if (final) {
    content = `<div class="auction-kicker">${e.winners.length > 1 ? "A SHARED VICTORY" : "THE WINNER"}</div><h3>${e.winners.map((id) => `Seat ${id + 1}`).join(" & ")} takes the table.</h3><div class="financial-card">${Math.max(...e.scores)} <span style="font-size:16px">points</span></div><p>All hidden hands revealed. All values settled.</p>`;
  } else {
    const won = `Seat ${e.winner + 1} won for ${e.bid} coin${e.bid === 1 ? "" : "s"}`;
    content = `<div class="auction-kicker">${e.type === "AUCTION" ? "TREASURE AUCTION" : e.type === "LOAN" ? "LIQUIDITY, AT A PRICE" : "COINS FOR TOMORROW"}</div><h3>${title(e)}</h3>${e.type === "AUCTION" ? `<div class="lot">${e.gems.map((g) => `<div class="gem-card">${gem(g, true)}</div>`).join("")}</div>` : `<div class="financial-card">${e.type === "LOAN" ? "↙" : "↗"} ${e.value}<span style="font-size:14px"> coins</span></div>`}<p>${won} ${e.reveal !== null ? `· Revealed ${colors[e.reveal]}` : "· No hidden gems left"}</p>${e.claimed.length ? `<p>✧ ${e.claimed.length} mission${e.claimed.length > 1 ? "s" : ""} claimed</p>` : ""}`;
  }
  $("auction-content").innerHTML = content;
  $("bids").innerHTML = s.players
    .map(
      (p) =>
        `<div class="bid-cell ${e?.winner === p.id ? "won" : ""}">Seat ${p.id + 1}<strong>${e?.bids ? e.bids[p.id] : final ? p.score : "—"}</strong>${e?.winner === p.id ? '<span class="winner-mark">WINNER ↗</span>' : ""}</div>`,
    )
    .join("");
  $("players").style.gridTemplateColumns =
    `repeat(${s.players.length},minmax(0,1fr))`;
  $("players").innerHTML = s.players
    .map(
      (p) =>
        `<article class="player ${e?.winner === p.id ? "active" : ""} ${final && e.winners.includes(p.id) ? "champion" : ""}"><div class="player-head"><span><i class="seat-dot s${p.id}"></i> SEAT ${p.id + 1}</span><span>${final && e.winners.includes(p.id) ? "WINNER" : e?.winner === p.id ? "LAST WINNER" : ""}</span></div><div class="bot-name">${escape(p.bot)}</div>${p.bot === "SoftTrackingBot" ? `<div class="bot-features">${escape(level3Features.filter(([key]) => replay.botOptions?.[p.id]?.[key]).map(([, label]) => label).join(" · ") || "Level 3 extras off")}</div>` : ""}<div class="money">${final ? p.score : p.coins} <small>${final ? "points" : "coins"}</small></div><div class="collection">${p.gems.some(Boolean) ? p.gems.map((count, g) => (count ? `<span class="gem-count">${gem(g)}${count}</span>` : "")).join("") : '<span class="empty">No gems collected yet</span>'}</div><div class="player-meta"><span>${p.missions.length} mission${p.missions.length === 1 ? "" : "s"}</span><span>${p.invested} locked coins</span></div><div class="hand"><span class="hand-label">${p.hand.length ? "HAND" : "REVEALED"}</span>${p.hand.length ? p.hand.map((g) => ($("show-hands").checked ? gem(g) : '<i class="back" title="Hidden information gem"></i>')).join("") : '<span class="empty">All public</span>'}</div>${
          final
            ? `<div class="breakdown">${Object.entries(p.breakdown)
                .map(
                  ([key, val]) =>
                    `<div><span>${key[0].toUpperCase() + key.slice(1)}</span><span>${val}</span></div>`,
                )
                .join("")}</div>`
            : ""
        }</article>`,
    )
    .join("");
  $("chart-label").textContent = `CHART ${replay.chart}`;
  $("market").innerHTML = colors
    .map(
      (name, g) =>
        `<div class="market-row">${gem(g)}<span class="color-name">${name}</span><span class="count">${s.display[g]} revealed</span><strong>${s.values[g]}<small> ¢</small></strong></div>`,
    )
    .join("");
  $("chart-table").innerHTML =
    `<thead><tr><th>Revealed</th>${replay.chartValues.map((_, i) => `<th>${i}</th>`).join("")}</tr></thead><tbody><tr><th>Points / gem</th>${replay.chartValues.map((v) => `<td>${v}</td>`).join("")}</tr></tbody>`;
  $("missions").innerHTML = replay.missions
    .map((m) =>
      missionMarkup(m, s.players.find((p) => p.missions.includes(m.id))?.id),
    )
    .join("");
  const past = replay.frames
    .slice(1, frame + 1)
    .map((f, i) => ({ ...f.event, round: i + 1 }))
    .filter((e) => e.type !== "FINAL")
    .reverse();
  $("journal-count").textContent =
    `${past.length} AUCTION${past.length === 1 ? "" : "S"}`;
  $("journal").innerHTML = past.length
    ? past
        .map(
          (e) =>
            `<div class="journal-row"><span class="journal-round">${String(e.round).padStart(2, "0")}</span><div><b>${title(e)} · Seat ${e.winner + 1} wins for ${e.bid}</b><div class="journal-detail">Bids: ${e.bids.map((b, i) => `S${i + 1}: ${b}`).join(" · ")}<br>${e.reveal === null ? "No hidden gem to reveal" : `Revealed ${colors[e.reveal]}`}${e.claimed.length ? ` · Claimed ${e.claimed.map((id) => `mission +${replay.missions[id].value}`).join(", ")}` : ""}</div></div></div>`,
        )
        .join("")
    : '<p class="muted">The table is set. The first auction is one step away.</p>';
}
$("seat-toggle").addEventListener("click", (e) => {
  const target = e.target.closest("[data-seats]");
  if (!target) return;
  seats = Number(target.dataset.seats);
  document
    .querySelectorAll("[data-seats]")
    .forEach((b) => b.classList.toggle("selected", b === target));
  seatFields();
});
$("setup").addEventListener("submit", async (e) => {
  e.preventDefault();
  stop();
  $("error").textContent = "";
  $("start").disabled = true;
  $("start").textContent = "Preparing the table…";
  document.body.classList.add("loading");
  try {
    const response = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        bots: [...document.querySelectorAll(".bot-select")].map((s) => s.value),
        botOptions: [...document.querySelectorAll(".seat-field")].map((field) =>
          field.querySelector(".bot-select").value === "SoftTrackingBot"
            ? Object.fromEntries([...field.querySelectorAll(".feature-switch")].map((el) => [el.dataset.feature, el.checked]))
            : {},
        ),
        chart: $("chart").value,
        seed: Number($("seed").value),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    replay = data;
    frame = 0;
    render();
  } catch (error) {
    $("error").textContent = error.message;
  } finally {
    $("start").disabled = false;
    $("start").innerHTML = "Start a new game <span>↗</span>";
    document.body.classList.remove("loading");
  }
});
$("play").addEventListener("click", play);
$("next").addEventListener("click", () => {
  stop();
  frame = Math.min(frame + 1, replay.frames.length - 1);
  render();
});
$("prev").addEventListener("click", () => {
  stop();
  frame = Math.max(0, frame - 1);
  render();
});
$("timeline").addEventListener("input", () => {
  stop();
  frame = Number($("timeline").value);
  render();
});
$("show-hands").addEventListener("change", render);
$("speed").addEventListener("change", () => {
  if (timer) {
    stop();
    play();
  }
});
async function init() {
  try {
    const response = await fetch("/api/bots");
    if (!response.ok) throw new Error("Cannot load bot catalog");
    catalog = await response.json();
    seatFields();
    $("setup").requestSubmit();
  } catch (error) {
    $("error").textContent = error.message;
  }
}
init();
