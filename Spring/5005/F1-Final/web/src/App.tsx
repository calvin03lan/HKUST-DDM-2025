import { useEffect, useMemo, useState } from "react";

type DriverCode = "ANT" | "RUS" | "PIA" | "NOR" | "LEC" | "HAM";

type DriverMeta = {
  name: string;
  team: string;
  number: string;
  color: string;
};

type Point = {
  t: number;
  lap: number;
  distance: number;
  totalDistance: number;
  speed: number;
  accel: number;
  compound: string;
  tyreLife: number;
};

type Scenario = {
  id: string;
  label: string;
  startLap: number;
  drivers: Record<DriverCode, Point[]>;
  result?: Array<{ position: number; driver: DriverCode; finishTime: number; gap: number }>;
};

type TrackPoint = { distance: number; x: number; y: number };

type PlaybackData = {
  drivers: Record<DriverCode, DriverMeta>;
  pitLaps: Record<DriverCode, number>;
  scLaps: number[];
  track: {
    length: number;
    laps: number;
    raceDistanceKm: number;
    cornerCount: number;
    path: TrackPoint[];
    corners: Array<{ id: number; distance: number; direction: string; speedClass: string; minSpeed: number }>;
  };
  scenarios: Scenario[];
  sensitivity: Array<{ piaInitialGapM: number; piaTireAdjustment: number; winner: DriverCode; gap: number }>;
};

type MetricsData = {
  breakdown: Array<{
    driver: DriverCode;
    absolute: Record<string, number>;
    relativeToANT: Record<string, number>;
  }>;
  tireIndex: Record<DriverCode, Record<string, Array<{ tyreLife: number; index: number }>>>;
};

type StoryData = {
  title: string;
  subtitle: string;
  timeline: Array<{ lap: string; title: string; body: string }>;
};

const driverOrder: DriverCode[] = ["ANT", "PIA", "RUS", "NOR", "LEC", "HAM"];

const teams = [
  {
    name: "Mercedes",
    car: "W17",
    trait: "长距离巡航稳定，清洁空气下后段速度强。",
    drivers: ["ANT", "RUS"] as DriverCode[],
  },
  {
    name: "McLaren",
    car: "MCL40",
    trait: "综合性能均衡，轮胎管理和中高速弯效率突出。",
    drivers: ["PIA", "NOR"] as DriverCode[],
  },
  {
    name: "Ferrari",
    car: "SF-26",
    trait: "起步和出弯加速强，低速弯牵引力有优势。",
    drivers: ["LEC", "HAM"] as DriverCode[],
  },
];

const driverNotes: Record<DriverCode, string> = {
  ANT: "最年轻杆位获得者之一，本场借安全车窗口完成 pole-to-win 式巡航。",
  RUS: "梅奔老将，负责压迫迈凯伦窗口并提前触发进站链条。",
  PIA: "本项目核心反事实对象，前 18 圈领跑并试图覆盖梅奔 undercut。",
  NOR: "卫冕冠军，提前进站为迈凯伦提供战术参照。",
  LEC: "法拉利绝对核心，弯道最低速和出弯牵引表现稳定。",
  HAM: "七届世界冠军，安全车下获得低损失进站机会。",
};

function App() {
  const [playback, setPlayback] = useState<PlaybackData | null>(null);
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [story, setStory] = useState<StoryData | null>(null);
  const [scenarioId, setScenarioId] = useState("actual");

  useEffect(() => {
    Promise.all([
      fetch("/data/telemetry_playback.json").then((res) => res.json()),
      fetch("/data/performance_metrics.json").then((res) => res.json()),
      fetch("/data/story_content.json").then((res) => res.json()),
    ]).then(([playbackData, metricsData, storyData]) => {
      setPlayback(playbackData);
      setMetrics(metricsData);
      setStory(storyData);
    });
  }, []);

  if (!playback || !metrics || !story) {
    return <div className="loading">正在加载铃鹿遥测数据...</div>;
  }

  const selectedScenario = playback.scenarios.find((item) => item.id === scenarioId) ?? playback.scenarios[0];

  return (
    <main>
      <Hero story={story} />
      <TrackIntro playback={playback} />
      <Teams drivers={playback.drivers} />
      <Timeline story={story} playback={playback} />
      <Hypotheses />
      <TechExplainer />
      <SimulationIntro playback={playback} />
      <TelemetryPlayer
        playback={playback}
        scenario={selectedScenario}
        scenarioId={scenarioId}
        onScenarioChange={setScenarioId}
      />
      <PerformanceBreakdown playback={playback} metrics={metrics} />
    </main>
  );
}

function Hero({ story }: { story: StoryData }) {
  return (
    <section className="hero">
      <div>
        <p className="eyebrow">Suzuka Game Theory · 2026 Japanese GP</p>
        <h1>{story.title}</h1>
        <p className="hero-subtitle">{story.subtitle}</p>
        <p className="hero-copy">
          皮亚斯特里第 18 圈进站覆盖梅奔，安东内利却在清洁空气和安全车窗口中完成逆转。本项目用遥测建模和反事实仿真回答：
          如果没有安全车，或者 PIA 更晚进站，他还有没有夺冠路径？
        </p>
      </div>
      <div className="hero-card">
        <span>核心问题</span>
        <strong>Safety Car or Strategy?</strong>
        <p>同一条赛道，三套数据：实际比赛、无安全车、PIA 晚进站。</p>
      </div>
    </section>
  );
}

function TrackIntro({ playback }: { playback: PlaybackData }) {
  const famous = [
    "First Turn",
    "S Curves",
    "Degner Curve",
    "Hairpin",
    "Spoon Curve",
    "Casio Triangle",
  ];
  return (
    <section className="panel track-intro">
      <div className="section-heading">
        <p className="eyebrow">Circuit</p>
        <h2>铃鹿赛道互动概览</h2>
        <p>以 FastF1 距离坐标和项目提供的赛道图为基础，标出弯角、著名赛段和 18 到 1 号弯之间的超车模式区域。</p>
      </div>
      <div className="track-layout">
        <div className="track-image-card">
          <img src="/assets/suzuka_straight_mode_zone.png" alt="Suzuka straight mode zone" />
          {playback.track.corners.slice(0, 18).map((corner, index) => (
            <span
              key={corner.id}
              className={`corner-pin corner-${index + 1}`}
              title={`T${corner.id} · ${corner.speedClass} · ${corner.minSpeed} km/h`}
            >
              {corner.id}
            </span>
          ))}
        </div>
        <div className="stat-grid">
          <Stat label="赛道长度" value={`${playback.track.length.toLocaleString()} m`} />
          <Stat label="总圈数" value={`${playback.track.laps} laps`} />
          <Stat label="比赛长度" value={`${playback.track.raceDistanceKm} km`} />
          <Stat label="弯角数" value={`${playback.track.cornerCount}`} />
          <div className="tag-list">
            {famous.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Teams({ drivers }: { drivers: Record<DriverCode, DriverMeta> }) {
  return (
    <section className="panel">
      <div className="section-heading">
        <p className="eyebrow">Teams & Drivers</p>
        <h2>三队六车手</h2>
        <p>素材缺失时先使用占位卡片，鼠标悬停可查看赛车特性和车手叙事标签。</p>
      </div>
      <div className="team-grid">
        {teams.map((team) => (
          <article className="team-card" key={team.name}>
            <div className="car-placeholder">{team.car}</div>
            <h3>{team.name}</h3>
            <p>{team.trait}</p>
            <div className="driver-pair">
              {team.drivers.map((code) => (
                <div className="driver-card" style={{ borderColor: drivers[code].color }} key={code}>
                  <span className="driver-number">#{drivers[code].number}</span>
                  <strong>{drivers[code].name}</strong>
                  <small>{code}</small>
                  <em>{driverNotes[code]}</em>
                </div>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function Timeline({ story, playback }: { story: StoryData; playback: PlaybackData }) {
  const finalOrder: DriverCode[] = ["ANT", "PIA", "RUS", "NOR", "LEC", "HAM"];
  return (
    <section className="panel timeline-panel">
      <div>
        <div className="section-heading">
          <p className="eyebrow">Race Recap</p>
          <h2>比赛关键节点</h2>
        </div>
        <div className="timeline">
          {story.timeline.map((item) => (
            <article key={item.lap}>
              <span>{item.lap}</span>
              <h3>{item.title}</h3>
              <p>{item.body}</p>
            </article>
          ))}
        </div>
      </div>
      <div className="ranking-card">
        <h3>实际完赛顺序</h3>
        {finalOrder.map((driver, index) => (
          <div className="rank-row" key={driver}>
            <b>{index + 1}</b>
            <span style={{ color: playback.drivers[driver].color }}>{driver}</span>
            <small>{playback.drivers[driver].name}</small>
          </div>
        ))}
      </div>
    </section>
  );
}

function Hypotheses() {
  return (
    <section className="hypothesis-grid">
      <article>
        <p className="eyebrow">Hypothesis 1</p>
        <h2>如果安全车没有出动？</h2>
        <p>PIA、RUS 已完成进站，ANT 和 HAM 必须在绿旗下付出完整进站损失。问题变成：PIA 能否利用已换白胎的位置收益重新接管领先？</p>
      </article>
      <article>
        <p className="eyebrow">Hypothesis 2</p>
        <h2>如果 PIA 更晚进站？</h2>
        <p>PIA 留在场上等待第 22 圈后的安全车窗口，并假设第 28 圈重启时仍处于 ANT 前方。后半程的轮胎窗口与清洁空气决定最终胜负。</p>
      </article>
    </section>
  );
}

function TechExplainer() {
  return (
    <section className="panel explainer">
      <div className="section-heading">
        <p className="eyebrow">Vehicle Model</p>
        <h2>为什么看加减速度和弯速</h2>
      </div>
      <div className="explain-grid">
        <article>
          <h3>加速度-速度曲线</h3>
          <p>同样油门开度下，赛车在不同速度区间的推进能力不同。直道末端速度越高，可用加速度通常越低。</p>
        </article>
        <article>
          <h3>减速度-速度曲线</h3>
          <p>制动能力决定入弯前能晚刹多少，也决定低速弯前的时间损失。高速段还会受到空气阻力影响。</p>
        </article>
        <article>
          <h3>低速弯</h3>
          <p>接近弯心时强制动到目标速度，再通过循迹刹车稳定车身，过弯心后线性开油。</p>
        </article>
        <article>
          <h3>中高速弯</h3>
          <p>以松油滑行和较小制动完成速度控制，尽量保持最低速度，直道后恢复最大加速度。</p>
        </article>
      </div>
    </section>
  );
}

function SimulationIntro({ playback }: { playback: PlaybackData }) {
  const winners = playback.scenarios.filter((scenario) => scenario.result?.length).map((scenario) => scenario.result![0]);
  return (
    <section className="panel sim-intro">
      <div className="section-heading">
        <p className="eyebrow">Simulation</p>
        <h2>仿真实验设置</h2>
        <p>赛车被简化为沿赛道距离推进的质点，状态包含圈数、距离、速度、加速度、胎种和胎龄；进站损失按实际战略窗口折算。</p>
      </div>
      <div className="winner-strip">
        {winners.map((winner) => (
          <div key={`${winner.driver}-${winner.finishTime}`}>
            <span>{winner.driver}</span>
            <strong>Winner</strong>
            <small>gap +{winner.gap}s</small>
          </div>
        ))}
      </div>
    </section>
  );
}

function TelemetryPlayer({
  playback,
  scenario,
  scenarioId,
  onScenarioChange,
}: {
  playback: PlaybackData;
  scenario: Scenario;
  scenarioId: string;
  onScenarioChange: (id: string) => void;
}) {
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);

  const duration = useMemo(() => {
    return Math.max(...driverOrder.flatMap((driver) => scenario.drivers[driver]?.map((point) => point.t) ?? [0]));
  }, [scenario]);

  useEffect(() => {
    setTime(0);
    setPlaying(false);
  }, [scenarioId]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setTime((current) => (current + 1 > duration ? 0 : current + 1));
    }, 160);
    return () => window.clearInterval(timer);
  }, [duration, playing]);

  const snapshots = driverOrder
    .map((driver) => {
      const point = interpolatePoint(scenario.drivers[driver] ?? [], time);
      return point ? { driver, point } : null;
    })
    .filter(Boolean) as Array<{ driver: DriverCode; point: Point }>;

  const ranked = [...snapshots].sort((a, b) => b.point.totalDistance - a.point.totalDistance);
  const leader = ranked[0];

  return (
    <section className="panel player-panel">
      <div className="section-heading">
        <p className="eyebrow">Telemetry Player</p>
        <h2>赛道数据播放器</h2>
        <p>选择场景后播放六位车手在铃鹿上的实时位置，排名由累计距离动态计算。</p>
      </div>
      <div className="player-controls">
        <select value={scenarioId} onChange={(event) => onScenarioChange(event.target.value)}>
          {playback.scenarios.map((item) => (
            <option key={item.id} value={item.id}>
              {item.label}
            </option>
          ))}
        </select>
        <button type="button" onClick={() => setPlaying((value) => !value)}>
          {playing ? "暂停" : "播放"}
        </button>
        <button type="button" onClick={() => setTime((value) => Math.max(0, value - 1))}>
          上一帧
        </button>
        <button type="button" onClick={() => setTime((value) => Math.min(duration, value + 1))}>
          下一帧
        </button>
        <label>
          时间 {Math.round(time)}s
          <input
            type="range"
            min={0}
            max={Math.max(1, Math.floor(duration))}
            value={Math.min(time, duration)}
            onChange={(event) => setTime(Number(event.target.value))}
          />
        </label>
      </div>
      <div className="player-grid">
        <div className="track-map">
          <svg viewBox="0 0 1000 620" role="img" aria-label="Telemetry track map">
            <polyline
              points={playback.track.path.map((point) => `${point.x * 900 + 50},${point.y * 520 + 50}`).join(" ")}
              className="track-line"
            />
            {snapshots.map(({ driver, point }) => {
              const pos = positionOnTrack(playback.track.path, point.distance);
              return (
                <g key={driver} transform={`translate(${pos.x * 900 + 50} ${pos.y * 520 + 50})`}>
                  <circle r="12" fill={playback.drivers[driver].color} />
                  <text y="4">{driver}</text>
                </g>
              );
            })}
          </svg>
        </div>
        <div className="live-ranking">
          <h3>实时排名</h3>
          <p>
            当前圈数：<strong>{leader?.point.lap ?? scenario.startLap}</strong>
          </p>
          {ranked.map(({ driver, point }, index) => {
            const gapM = leader ? leader.point.totalDistance - point.totalDistance : 0;
            return (
              <div className="rank-row telemetry-row" key={driver}>
                <b>{index + 1}</b>
                <span style={{ color: playback.drivers[driver].color }}>{driver}</span>
                <small>
                  {point.compound} · {Math.round(point.speed)} km/h · +{Math.max(0, gapM).toFixed(0)} m
                </small>
              </div>
            );
          })}
          {scenario.result && (
            <div className="scenario-result">
              <strong>终点结果</strong>
              {scenario.result.slice(0, 3).map((item) => (
                <span key={item.driver}>
                  P{item.position} {item.driver} +{item.gap}s
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function interpolatePoint(points: Point[], time: number): Point | null {
  if (!points.length) return null;
  if (time <= points[0].t) return points[0];
  for (let index = 1; index < points.length; index += 1) {
    const prev = points[index - 1];
    const next = points[index];
    if (next.t >= time) {
      const ratio = (time - prev.t) / Math.max(0.001, next.t - prev.t);
      return {
        ...next,
        t: time,
        lap: Math.round(prev.lap + (next.lap - prev.lap) * ratio),
        distance: prev.distance + (next.distance - prev.distance) * ratio,
        totalDistance: prev.totalDistance + (next.totalDistance - prev.totalDistance) * ratio,
        speed: prev.speed + (next.speed - prev.speed) * ratio,
        accel: prev.accel + (next.accel - prev.accel) * ratio,
      };
    }
  }
  return points[points.length - 1];
}

function positionOnTrack(path: TrackPoint[], distance: number) {
  if (!path.length) return { x: 0.5, y: 0.5 };
  const normalizedDistance = ((distance % 5807) + 5807) % 5807;
  for (let index = 1; index < path.length; index += 1) {
    const prev = path[index - 1];
    const next = path[index];
    if (next.distance >= normalizedDistance) {
      const ratio = (normalizedDistance - prev.distance) / Math.max(0.001, next.distance - prev.distance);
      return {
        x: prev.x + (next.x - prev.x) * ratio,
        y: prev.y + (next.y - prev.y) * ratio,
      };
    }
  }
  return path[path.length - 1];
}

function PerformanceBreakdown({ playback, metrics }: { playback: PlaybackData; metrics: MetricsData }) {
  const labels: Record<string, string> = {
    maxAcceleration: "最大加速度",
    maxDeceleration: "最大减速度",
    lowCorner: "低速弯",
    mediumCorner: "中速弯",
    highCorner: "高速弯",
    racePace: "长距离节奏",
  };
  return (
    <section className="panel breakdown">
      <div className="section-heading">
        <p className="eyebrow">Performance Breakdown</p>
        <h2>以 ANT 为基准的表现分解</h2>
        <p>数值为相对 ANT 的表现指数，1.00 表示与 ANT 基准一致；长距离节奏越低越好，因此已做反向归一化。</p>
      </div>
      <div className="breakdown-grid">
        {metrics.breakdown.map((row) => (
          <article className="metric-card" key={row.driver}>
            <h3 style={{ color: playback.drivers[row.driver].color }}>
              {row.driver} <small>{playback.drivers[row.driver].name}</small>
            </h3>
            {Object.entries(labels).map(([key, label]) => {
              const value = row.relativeToANT[key] ?? 1;
              return (
                <div className="metric-row" key={key}>
                  <span>{label}</span>
                  <div>
                    <i style={{ width: `${Math.min(130, value * 100)}%`, background: playback.drivers[row.driver].color }} />
                  </div>
                  <b>{value.toFixed(3)}</b>
                </div>
              );
            })}
          </article>
        ))}
      </div>
      <div className="sensitivity">
        <h3>敏感性分析摘要</h3>
        {playback.sensitivity.slice(0, 8).map((row) => (
          <span key={`${row.piaInitialGapM}-${row.piaTireAdjustment}`}>
            PIA +{row.piaInitialGapM}m / 胎耗 {row.piaTireAdjustment > 0 ? "+" : ""}
            {row.piaTireAdjustment}: {row.winner} 赢 {row.gap}s
          </span>
        ))}
      </div>
    </section>
  );
}

export default App;
