import { useEffect, useMemo, useState } from "react";

import {
  Activity,
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Database,
  GitBranch,
  LayoutDashboard,
  Network,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  Target,
  X,
  Zap,
} from "lucide-react";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
} from "@xyflow/react";

import "@xyflow/react/dist/style.css";
import "./App.css";


/* ============================================================
   TRACE-X API
   ============================================================ */

const API_BASE = (
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8001"
).replace(/\/$/, "");


/* ============================================================
   DEFAULT DATA
   ============================================================ */

const EMPTY_SUMMARY = {
  total_transactions: 0,
  flagged_transactions: 0,
  normal_transactions: 0,
  laundering_rate: 0,
  total_received: 0,
  total_paid: 0,
  average_transaction: 0,
};

const DEFAULT_TRANSACTION = {
  timestamp: "2022-09-01 00:02:00",
  from_bank: "070",
  sender_account: "10042B660",
  to_bank: "022661",
  receiver_account: "805F7F2B0",
  amount_received: 70831.64,
  receiving_currency: "US Dollar",
  amount_paid: 70831.64,
  payment_currency: "US Dollar",
  payment_format: "Cash",
};


/* ============================================================
   FORMATTERS
   ============================================================ */

function formatNumber(value) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(Number(value ?? 0));
}

function formatMoney(value) {
  const n = Number(value ?? 0);

  if (n >= 1_000_000_000_000) {
    return `$${(n / 1_000_000_000_000).toFixed(2)}T`;
  }

  if (n >= 1_000_000_000) {
    return `$${(n / 1_000_000_000).toFixed(2)}B`;
  }

  if (n >= 1_000_000) {
    return `$${(n / 1_000_000).toFixed(2)}M`;
  }

  if (n >= 1_000) {
    return `$${(n / 1_000).toFixed(1)}K`;
  }

  return `$${n.toFixed(2)}`;
}

function formatPercent(value) {
  return `${(Number(value ?? 0) * 100).toFixed(2)}%`;
}


/* ============================================================
   API HELPER
   ============================================================ */

async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    cache: "no-store",
    ...options,
    headers: {
      Accept: "application/json",
      ...(options.headers || {}),
    },
  });

  const text = await response.text();

  let data = null;

  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }

  if (!response.ok) {
    throw new Error(
      data?.detail ||
        data?.message ||
        `HTTP ${response.status}`
    );
  }

  return data;
}


/* ============================================================
   RESPONSE HELPERS
   ============================================================ */

function extractRows(data) {
  if (Array.isArray(data)) {
    return data;
  }

  if (Array.isArray(data?.data)) {
    return data.data;
  }

  if (Array.isArray(data?.items)) {
    return data.items;
  }

  if (Array.isArray(data?.results)) {
    return data.results;
  }

  return [];
}

function normalizeSummary(data) {
  const source =
    data?.summary ||
    data?.data ||
    data ||
    {};

  return {
    total_transactions: Number(
      source.total_transactions ?? 0
    ),

    flagged_transactions: Number(
      source.flagged_transactions ?? 0
    ),

    normal_transactions: Number(
      source.normal_transactions ?? 0
    ),

    laundering_rate: Number(
      source.laundering_rate ?? 0
    ),

    total_received: Number(
      source.total_received ?? 0
    ),

    total_paid: Number(
      source.total_paid ?? 0
    ),

    average_transaction: Number(
      source.average_transaction ?? 0
    ),
  };
}


/* ============================================================
   APP
   ============================================================ */

export default function App() {
  const [page, setPage] =
    useState("command");

  const [summary, setSummary] =
    useState(EMPTY_SUMMARY);

  const [model, setModel] =
    useState(null);

  const [trends, setTrends] =
    useState([]);

  const [distribution, setDistribution] =
    useState([]);

  const [banks, setBanks] =
    useState([]);

  const [queue, setQueue] =
    useState([]);

  const [network, setNetwork] =
    useState({
      nodes: [],
      edges: [],
    });

  const [status, setStatus] =
    useState("CONNECTING");

  const [error, setError] =
    useState("");

  const [loading, setLoading] =
    useState(false);

  const [search, setSearch] =
    useState("");

  const [bankFilter, setBankFilter] =
    useState("");

  const [selectedTransaction, setSelectedTransaction] =
    useState(null);

  const [showAnalyzer, setShowAnalyzer] =
    useState(false);

  const [form, setForm] =
    useState(DEFAULT_TRANSACTION);

  const [prediction, setPrediction] =
    useState(null);

  // GenAI investigation state.
  // This is intentionally separate from the existing TRACE-X
  // prediction state so the frozen model workflow is untouched.
  const [aiInvestigation, setAiInvestigation] =
    useState(null);


  /* ==========================================================
     CORE DATA
     ========================================================== */

  async function refreshCore() {
    setLoading(true);
    setError("");

    try {
      const results =
        await Promise.allSettled([
          apiRequest("/health"),
          apiRequest(
            "/api/v1/dashboard/summary"
          ),
          apiRequest("/api/v1/model"),
        ]);

      const healthResult = results[0];
      const summaryResult = results[1];
      const modelResult = results[2];

      if (
        healthResult.status ===
        "rejected"
      ) {
        throw new Error(
          `Backend health check failed: ${
            healthResult.reason?.message ||
            healthResult.reason
          }`
        );
      }

      const health =
        healthResult.value;

      setStatus(
        health?.status === "healthy"
          ? "OPERATIONAL"
          : "DEGRADED"
      );

      if (
        summaryResult.status ===
        "rejected"
      ) {
        throw new Error(
          `Dashboard summary failed: ${
            summaryResult.reason?.message ||
            summaryResult.reason
          }`
        );
      }

      const cleanSummary =
        normalizeSummary(
          summaryResult.value
        );

      setSummary(cleanSummary);

      if (
        modelResult.status ===
        "fulfilled"
      ) {
        setModel(modelResult.value);
      }
    } catch (err) {
      console.error(
        "TRACE-X core API error:",
        err
      );

      setStatus("OFFLINE");

      setError(
        err?.message ||
          "Unable to connect to TRACE-X backend."
      );
    } finally {
      setLoading(false);
    }


    /* ----------------------------------------------------------
       OPTIONAL DATA
       ---------------------------------------------------------- */

    const optionalEndpoints = [
      [
        "/api/v1/dashboard/trends",
        setTrends,
      ],
      [
        "/api/v1/dashboard/distribution",
        setDistribution,
      ],
      [
        "/api/v1/dashboard/top-banks",
        setBanks,
      ],
      [
        "/api/v1/dashboard/queue?limit=100",
        setQueue,
      ],
    ];

    await Promise.all(
      optionalEndpoints.map(
        async ([endpoint, setter]) => {
          try {
            const data =
              await apiRequest(endpoint);

            setter(
              extractRows(data)
            );
          } catch (err) {
            console.warn(
              `Optional endpoint failed: ${endpoint}`,
              err
            );

            setter([]);
          }
        }
      )
    );
  }


  /* ==========================================================
     NETWORK
     ========================================================== */

  async function loadNetwork() {
    try {
      const query =
        bankFilter.trim()
          ? `?bank=${encodeURIComponent(
              bankFilter.trim()
            )}&limit=60`
          : "?limit=60";

      const data =
        await apiRequest(
          `/api/v1/dashboard/network${query}`
        );

      setNetwork({
        nodes: Array.isArray(
          data?.nodes
        )
          ? data.nodes
          : [],

        edges: Array.isArray(
          data?.edges
        )
          ? data.edges
          : [],
      });
    } catch (err) {
      console.warn(
        "Network endpoint failed:",
        err
      );

      setNetwork({
        nodes: [],
        edges: [],
      });
    }
  }


  /* ==========================================================
     INITIAL LOAD
     ========================================================== */

  useEffect(() => {
    refreshCore();
  }, []);


  useEffect(() => {
    if (page === "network") {
      loadNetwork();
    }
  }, [page, bankFilter]);


  /* ==========================================================
     TRANSACTION ANALYSIS
     ========================================================== */

  async function analyzeTransaction() {
    setPrediction({
      loading: true,
    });

    try {
      const payload = {
        timestamp: String(
          form.timestamp
        ),

        from_bank: String(
          form.from_bank
        ),

        sender_account: String(
          form.sender_account
        ),

        to_bank: String(
          form.to_bank
        ),

        receiver_account: String(
          form.receiver_account
        ),

        amount_received: Number(
          form.amount_received
        ),

        receiving_currency: String(
          form.receiving_currency
        ),

        amount_paid: Number(
          form.amount_paid
        ),

        payment_currency: String(
          form.payment_currency
        ),

        payment_format: String(
          form.payment_format
        ),
      };

      const data =
        await apiRequest(
          "/api/v1/predict/transaction",
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify(payload),
          }
        );

      setPrediction(data);
    } catch (err) {
      console.error(
        "TRACE-X prediction:",
        err
      );

      setPrediction({
        error:
          err?.message ||
          "Transaction analysis failed.",
      });
    }
  }


  /* ==========================================================
     GENAI INVESTIGATION
     ========================================================== */

  async function runAIInvestigation() {
    setAiInvestigation({
      loading: true,
      error: "",
      ai: null,
      trace_x: null,
      historical_features: null,
    });

    try {
      const payload = {
        timestamp: String(
          form.timestamp
        ),

        from_bank: String(
          form.from_bank
        ),

        sender_account: String(
          form.sender_account
        ),

        to_bank: String(
          form.to_bank
        ),

        receiver_account: String(
          form.receiver_account
        ),

        amount_received: Number(
          form.amount_received
        ),

        receiving_currency: String(
          form.receiving_currency
        ),

        amount_paid: Number(
          form.amount_paid
        ),

        payment_currency: String(
          form.payment_currency
        ),

        payment_format: String(
          form.payment_format
        ),
      };

      const data =
        await apiRequest(
          "/api/v1/ai/investigate",
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify(payload),
          }
        );

      setAiInvestigation({
        ...data,
        loading: false,
        error: "",
      });
    } catch (err) {
      console.error(
        "TRACE-X GenAI investigation:",
        err
      );

      setAiInvestigation({
        loading: false,
        error:
          err?.message ||
          "AI investigation failed.",
        ai: null,
        trace_x: null,
        historical_features: null,
      });
    }
  }


  /* ==========================================================
     QUEUE FILTER
     ========================================================== */

  const filteredQueue =
    useMemo(() => {
      const q =
        search
          .trim()
          .toLowerCase();

      if (!q) {
        return queue;
      }

      return queue.filter(
        (item) =>
          Object.values(item)
            .join(" ")
            .toLowerCase()
            .includes(q)
      );
    }, [queue, search]);


  /* ==========================================================
     DISTRIBUTION
     ========================================================== */

  const chartDistribution =
    useMemo(() => {
      const rows =
        extractRows(
          distribution
        )
          .map((item) => ({
            name: String(
              item.decision ??
                item.label ??
                item.name ??
                "UNKNOWN"
            ),

            value: Number(
              item.count ??
                item.value ??
                item.total ??
                0
            ),
          }))
          .filter(
            (item) =>
              item.value > 0
          );

      if (rows.length) {
        return rows;
      }

      return [
        {
          name: "NORMAL",
          value:
            summary.normal_transactions,
        },

        {
          name: "FLAG",
          value:
            summary.flagged_transactions,
        },
      ].filter(
        (item) =>
          item.value > 0
      );
    }, [
      distribution,
      summary,
    ]);


  /* ==========================================================
     TRENDS
     ========================================================== */

  const chartTrends =
    useMemo(() => {
      return trends.map(
        (item) => ({
          period:
            item.period ??
            item.timestamp ??
            item.date ??
            item.hour ??
            "",

          transactions: Number(
            item.transactions ??
              item.total ??
              item.count ??
              0
          ),

          flagged: Number(
            item.flagged ??
              item.flagged_transactions ??
              item.flags ??
              0
          ),
        })
      );
    }, [trends]);


  /* ==========================================================
     REACT FLOW
     ========================================================== */

  const flowNodes =
    useMemo(() => {
      return network.nodes.map(
        (node, index) => ({
          id: String(
            node.id ??
              node.name ??
              index
          ),

          data: {
            label:
              node.label ??
              node.name ??
              node.id ??
              `Node ${index + 1}`,
          },

          position: {
            x:
              (index % 5) * 210,

            y:
              Math.floor(
                index / 5
              ) * 140,
          },

          style: {
            background:
              "#07101f",

            color:
              "#e2e8f0",

            border:
              "1px solid rgba(34,211,238,.4)",

            borderRadius: 12,

            padding: 12,

            width: 155,

            fontSize: 11,
          },
        })
      );
    }, [network.nodes]);


  const flowEdges =
    useMemo(() => {
      return network.edges.map(
        (edge, index) => ({
          id: String(
            edge.id ??
              `edge-${index}`
          ),

          source: String(
            edge.source
          ),

          target: String(
            edge.target
          ),

          animated: true,

          style: {
            stroke:
              "#22d3ee",

            strokeWidth: 1.5,
          },
        })
      );
    }, [network.edges]);


  /* ==========================================================
     HELPERS
     ========================================================== */

  function openAnalyzer(
    transaction = null
  ) {
    if (transaction) {
      setForm({
        ...DEFAULT_TRANSACTION,
        ...transaction,

        from_bank: String(
          transaction.from_bank ??
            DEFAULT_TRANSACTION.from_bank
        ),

        to_bank: String(
          transaction.to_bank ??
            DEFAULT_TRANSACTION.to_bank
        ),

        amount_received: Number(
          transaction.amount_received ??
            0
        ),

        amount_paid: Number(
          transaction.amount_paid ??
            0
        ),
      });
    } else {
      setForm(
        DEFAULT_TRANSACTION
      );
    }

    setPrediction(null);
    setAiInvestigation(null);
    setShowAnalyzer(true);
  }


  const pageTitle = {
    command:
      "Command Center",

    queue:
      "Investigation Queue",

    network:
      "Network Intelligence",

    rules:
      "Rule Engine",

    ai:
      "AI Investigation",
  }[page];


  /* ==========================================================
     RENDER
     ========================================================== */

  return (
    <div className="trace-shell">

      <aside className="trace-sidebar">

        <div className="trace-logo">

          <div className="logo-core">
            <ShieldCheck size={25} />
          </div>

          <div>
            <div className="logo-title">
              TRACE-X
            </div>

            <div className="logo-subtitle">
              FINANCIAL INTELLIGENCE
            </div>
          </div>

        </div>


        <div className="system-pill">

          <span
            className={
              status ===
              "OPERATIONAL"
                ? "status-dot online"
                : "status-dot"
            }
          />

          <span>
            SYSTEM {status}
          </span>

        </div>


        <nav className="trace-nav">

          <NavItem
            icon={
              <LayoutDashboard
                size={18}
              />
            }
            label="Command Center"
            active={
              page === "command"
            }
            onClick={() =>
              setPage("command")
            }
          />

          <NavItem
            icon={
              <AlertTriangle
                size={18}
              />
            }
            label="Investigation Queue"
            badge={
              summary.flagged_transactions
            }
            active={
              page === "queue"
            }
            onClick={() =>
              setPage("queue")
            }
          />

          <NavItem
            icon={
              <Network
                size={18}
              />
            }
            label="Network Intelligence"
            active={
              page === "network"
            }
            onClick={() =>
              setPage("network")
            }
          />

          <NavItem
            icon={
              <GitBranch
                size={18}
              />
            }
            label="Rule Engine"
            active={
              page === "rules"
            }
            onClick={() =>
              setPage("rules")
            }
          />

          <NavItem
            icon={
              <BrainCircuit
                size={18}
              />
            }
            label="AI Investigation"
            active={
              page === "ai"
            }
            onClick={() =>
              setPage("ai")
            }
          />

        </nav>


        <div className="sidebar-bottom">

          <div className="model-card">

            <div className="model-card-header">
              <Zap size={15} />
              TRACE-X V1
            </div>

            <ModelLine
              label="MODEL"
              value={
                model?.model ??
                "FROZEN"
              }
            />

            <ModelLine
              label="ROUNDS"
              value={
                model?.boosting_rounds ??
                800
              }
            />

            <ModelLine
              label="FEATURES"
              value={
                model?.feature_count ??
                38
              }
            />

            <ModelLine
              label="THRESHOLD"
              value={
                model?.threshold ??
                0.76
              }
            />

          </div>

        </div>

      </aside>


      <main className="trace-main">

        <header className="trace-header">

          <div>

            <div className="eyebrow">
              <CircleDot size={11} />
              FINANCIAL CRIME INTELLIGENCE
            </div>

            <h1>
              {pageTitle}
            </h1>

          </div>


          <div className="header-actions">

            <button
              className="icon-button"
              type="button"
              title="Refresh"
              onClick={
                refreshCore
              }
            >
              <RefreshCw
                size={18}
                className={
                  loading
                    ? "spin"
                    : ""
                }
              />
            </button>


            <button
              className="primary-button"
              type="button"
              onClick={() =>
                openAnalyzer()
              }
            >
              <Sparkles size={17} />
              Analyze Transaction
            </button>

          </div>

        </header>


        {error && (
          <div
            className="error-box"
            style={{
              marginBottom: 16,
            }}
          >
            <strong>
              TRACE-X backend connection error
            </strong>

            <br />

            {error}
          </div>
        )}


        {page === "command" && (
          <CommandPage
            summary={summary}
            trends={chartTrends}
            distribution={
              chartDistribution
            }
            banks={banks}
            queue={queue}
            onSelect={
              setSelectedTransaction
            }
          />
        )}


        {page === "queue" && (
          <QueuePage
            queue={
              filteredQueue
            }
            search={search}
            setSearch={
              setSearch
            }
            onSelect={
              setSelectedTransaction
            }
          />
        )}


        {page === "network" && (
          <NetworkPage
            nodes={flowNodes}
            edges={flowEdges}
            bank={
              bankFilter
            }
            setBank={
              setBankFilter
            }
          />
        )}


        {page === "rules" && (
          <RulesPage
            model={model}
            summary={summary}
          />
        )}


        {page === "ai" && (
          <AIPage
            prediction={
              prediction
            }
            aiInvestigation={
              aiInvestigation
            }
            onAnalyze={() =>
              openAnalyzer()
            }
            onRunAI={
              runAIInvestigation
            }
          />
        )}

      </main>


      {selectedTransaction && (
        <CaseDrawer
          transaction={
            selectedTransaction
          }
          onClose={() =>
            setSelectedTransaction(
              null
            )
          }
          onAnalyze={() => {
            setSelectedTransaction(
              null
            );

            openAnalyzer(
              selectedTransaction
            );
          }}
        />
      )}


      {showAnalyzer && (
        <Analyzer
          form={form}
          setForm={setForm}
          prediction={
            prediction
          }
          onAnalyze={
            analyzeTransaction
          }
          onClose={() =>
            setShowAnalyzer(
              false
            )
          }
        />
      )}

    </div>
  );
}


/* ============================================================
   NAV
   ============================================================ */

function NavItem({
  icon,
  label,
  badge,
  active,
  onClick,
}) {
  return (
    <button
      type="button"
      className={
        active
          ? "nav-button active"
          : "nav-button"
      }
      onClick={onClick}
    >
      {icon}

      <span>
        {label}
      </span>

      {Number(badge) > 0 && (
        <span className="nav-badge">
          {formatNumber(badge)}
        </span>
      )}

      {active && (
        <ChevronRight
          size={15}
          className="nav-arrow"
        />
      )}
    </button>
  );
}


function ModelLine({
  label,
  value,
}) {
  return (
    <div className="model-status">
      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>
    </div>
  );
}


/* ============================================================
   COMMAND PAGE
   ============================================================ */

function CommandPage({
  summary,
  trends,
  distribution,
  banks,
  queue,
  onSelect,
}) {
  return (
    <div className="page-content">

      <div className="metric-grid">

        <Metric
          icon={<Database />}
          label="Transactions Analyzed"
          value={formatNumber(
            summary.total_transactions
          )}
          meta="Authoritative dataset"
        />

        <Metric
          icon={<AlertTriangle />}
          label="Flagged Transactions"
          value={formatNumber(
            summary.flagged_transactions
          )}
          meta={formatPercent(
            summary.laundering_rate
          )}
          danger
        />

        <Metric
          icon={<CheckCircle2 />}
          label="Normal Transactions"
          value={formatNumber(
            summary.normal_transactions
          )}
          meta="Dataset-cleared population"
        />

        <Metric
          icon={<Target />}
          label="Average Transaction"
          value={formatMoney(
            summary.average_transaction
          )}
          meta="Amount received"
        />

      </div>


      <div className="dashboard-grid">

        <section className="panel chart-panel large">

          <PanelHeader
            icon={
              <Activity
                size={17}
              />
            }
            title="Transaction Activity"
            subtitle="Observed transaction activity from TRACE-X analytics"
          />

          <div className="chart-container">

            {trends.length > 0 ? (
              <ResponsiveContainer
                width="100%"
                height="100%"
              >
                <AreaChart
                  data={trends}
                >

                  <defs>
                    <linearGradient
                      id="traceArea"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop
                        offset="0%"
                        stopColor="#22d3ee"
                        stopOpacity={0.42}
                      />

                      <stop
                        offset="100%"
                        stopColor="#22d3ee"
                        stopOpacity={0}
                      />
                    </linearGradient>
                  </defs>

                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(148,163,184,.08)"
                  />

                  <XAxis
                    dataKey="period"
                    tick={{
                      fill:
                        "#64748b",
                      fontSize: 9,
                    }}
                  />

                  <YAxis
                    tick={{
                      fill:
                        "#64748b",
                      fontSize: 9,
                    }}
                  />

                  <Tooltip />

                  <Area
                    type="monotone"
                    dataKey="transactions"
                    stroke="#22d3ee"
                    fill="url(#traceArea)"
                    strokeWidth={2}
                  />

                  <Area
                    type="monotone"
                    dataKey="flagged"
                    stroke="#fb7185"
                    fill="transparent"
                    strokeWidth={2}
                  />

                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <EmptyChart
                title="No trend series returned"
                subtitle="Core dashboard totals are shown above."
              />
            )}

          </div>

        </section>


        <section className="panel chart-panel">

          <PanelHeader
            icon={
              <BarChart3
                size={17}
              />
            }
            title="Decision Distribution"
            subtitle="Actual dataset classification"
          />

          <div className="donut-container">

            <ResponsiveContainer
              width="100%"
              height="100%"
            >
              <PieChart>

                <Pie
                  data={distribution}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={65}
                  outerRadius={105}
                  paddingAngle={3}
                >

                  {distribution.map(
                    (entry, index) => (
                      <Cell
                        key={`${entry.name}-${index}`}
                        fill={
                          String(
                            entry.name
                          ).toUpperCase() ===
                          "FLAG"
                            ? "#fb7185"
                            : "#22d3ee"
                        }
                      />
                    )
                  )}

                </Pie>

                <Tooltip />

              </PieChart>
            </ResponsiveContainer>


            <div className="donut-center">

              <strong>
                {formatPercent(
                  summary.laundering_rate
                )}
              </strong>

              <span>
                FLAG RATE
              </span>

            </div>

          </div>

        </section>


        <section className="panel chart-panel">

          <PanelHeader
            icon={
              <Target
                size={17}
              />
            }
            title="Bank Risk Signals"
            subtitle="Flagged activity by bank"
          />

          <div className="chart-container">

            {banks.length > 0 ? (
              <ResponsiveContainer
                width="100%"
                height="100%"
              >

                <BarChart
                  data={banks}
                  layout="vertical"
                  margin={{
                    left: 10,
                    right: 15,
                  }}
                >

                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(148,163,184,.08)"
                  />

                  <XAxis
                    type="number"
                    tick={{
                      fill:
                        "#64748b",
                      fontSize: 9,
                    }}
                  />

                  <YAxis
                    type="category"
                    dataKey="bank"
                    width={55}
                    tick={{
                      fill:
                        "#94a3b8",
                      fontSize: 9,
                    }}
                  />

                  <Tooltip />

                  <Bar
                    dataKey="flagged"
                    fill="#fb7185"
                    radius={[
                      0,
                      5,
                      5,
                      0,
                    ]}
                  />

                </BarChart>

              </ResponsiveContainer>
            ) : (
              <EmptyChart
                title="No bank series returned"
                subtitle="Core model remains operational."
              />
            )}

          </div>

        </section>


        <section className="panel queue-panel">

          <PanelHeader
            icon={
              <AlertTriangle
                size={17}
              />
            }
            title="Investigation Queue"
            subtitle="Flagged transactions requiring review"
          />

          <TransactionTable
            queue={queue.slice(
              0,
              8
            )}
            onSelect={
              onSelect
            }
          />

        </section>

      </div>

    </div>
  );
}


/* ============================================================
   METRIC
   ============================================================ */

function Metric({
  icon,
  label,
  value,
  meta,
  danger = false,
}) {
  return (
    <div className="metric-card">

      <div
        className={
          danger
            ? "metric-icon danger"
            : "metric-icon"
        }
      >
        {icon}
      </div>

      <div className="metric-body">

        <span>
          {label}
        </span>

        <strong>
          {value}
        </strong>

        <small>
          {meta}
        </small>

      </div>

    </div>
  );
}


/* ============================================================
   PANEL HEADER
   ============================================================ */

function PanelHeader({
  icon,
  title,
  subtitle,
}) {
  return (
    <div className="panel-header">

      <div className="panel-title">

        <div className="panel-icon">
          {icon}
        </div>

        <div>

          <h3>
            {title}
          </h3>

          <p>
            {subtitle}
          </p>

        </div>

      </div>

    </div>
  );
}


/* ============================================================
   EMPTY
   ============================================================ */

function EmptyChart({
  title,
  subtitle,
}) {
  return (
    <div className="empty-chart">

      <Activity size={26} />

      <strong>
        {title}
      </strong>

      <span>
        {subtitle}
      </span>

    </div>
  );
}


/* ============================================================
   QUEUE
   ============================================================ */

function QueuePage({
  queue,
  search,
  setSearch,
  onSelect,
}) {
  return (
    <div className="page-content">

      <div className="toolbar">

        <div className="search-box">

          <Search size={17} />

          <input
            value={search}
            onChange={(e) =>
              setSearch(
                e.target.value
              )
            }
            placeholder="Search accounts, banks, currency..."
          />

        </div>

        <div className="toolbar-stat">
          {formatNumber(
            queue.length
          )}{" "}
          visible cases
        </div>

      </div>


      <section className="panel full-panel">

        <PanelHeader
          icon={
            <AlertTriangle
              size={17}
            />
          }
          title="Investigation Queue"
          subtitle="Select a transaction to inspect its evidence"
        />

        <TransactionTable
          queue={queue}
          onSelect={
            onSelect
          }
        />

      </section>

    </div>
  );
}


/* ============================================================
   TRANSACTION TABLE
   ============================================================ */

function TransactionTable({
  queue,
  onSelect,
}) {
  return (
    <div className="table-wrap">

      <table>

        <thead>
          <tr>
            <th>TIME</th>
            <th>SENDER</th>
            <th>RECEIVER</th>
            <th>AMOUNT</th>
            <th>FORMAT</th>
            <th>STATUS</th>
            <th />
          </tr>
        </thead>

        <tbody>

          {queue.length === 0 ? (
            <tr>
              <td
                colSpan="7"
                className="empty-state"
              >
                No flagged transactions available.
              </td>
            </tr>
          ) : (
            queue.map(
              (item, index) => (
                <tr
                  key={
                    item.id ??
                    `${item.timestamp}-${index}`
                  }
                  onClick={() =>
                    onSelect(item)
                  }
                >

                  <td>
                    {item.timestamp ??
                      "—"}
                  </td>

                  <td>
                    <span className="mono">
                      {item.from_bank ??
                        "—"}
                    </span>

                    <small>
                      {item.sender_account ??
                        "—"}
                    </small>
                  </td>

                  <td>
                    <span className="mono">
                      {item.to_bank ??
                        "—"}
                    </span>

                    <small>
                      {item.receiver_account ??
                        "—"}
                    </small>
                  </td>

                  <td>
                    {formatMoney(
                      item.amount_received
                    )}
                  </td>

                  <td>
                    {item.payment_format ??
                      "—"}
                  </td>

                  <td>
                    <span className="flag-pill">
                      FLAG
                    </span>
                  </td>

                  <td>
                    <ChevronRight
                      size={16}
                    />
                  </td>

                </tr>
              )
            )
          )}

        </tbody>

      </table>

    </div>
  );
}


/* ============================================================
   NETWORK
   ============================================================ */

function NetworkPage({
  nodes,
  edges,
  bank,
  setBank,
}) {
  return (
    <div className="page-content">

      <div className="toolbar">

        <div className="search-box">

          <Network size={17} />

          <input
            value={bank}
            onChange={(e) =>
              setBank(
                e.target.value
              )
            }
            placeholder="Filter network by bank..."
          />

        </div>

        <div className="toolbar-stat">
          {nodes.length} nodes ·{" "}
          {edges.length} edges
        </div>

      </div>


      <section className="panel network-panel">

        <PanelHeader
          icon={
            <Network
              size={17}
            />
          }
          title="Transaction Network"
          subtitle="Relationship intelligence across transaction entities"
        />

        <div className="flow-container">

          {nodes.length === 0 ? (
            <EmptyChart
              title="Network data unavailable"
              subtitle="No network series was returned."
            />
          ) : (
            <ReactFlow
              nodes={nodes}
              edges={edges}
              fitView
              proOptions={{
                hideAttribution: true,
              }}
            >

              <Background
                color="#1e293b"
                gap={24}
              />

              <Controls />

              <MiniMap
                nodeColor="#22d3ee"
              />

            </ReactFlow>
          )}

        </div>

      </section>

    </div>
  );
}


/* ============================================================
   RULES
   ============================================================ */

function RulesPage({
  model,
  summary,
}) {
  return (
    <div className="page-content">

      <section className="hero-panel">

        <div className="hero-icon">
          <GitBranch size={30} />
        </div>

        <div>

          <div className="eyebrow">
            TRACE-X DECISION ENGINE
          </div>

          <h2>
            Detection Architecture
          </h2>

          <p>
            Frozen model inference combined
            with historical transaction intelligence.
          </p>

        </div>

      </section>


      <div className="rule-grid">

        <RuleCard
          title="MODEL"
          value={
            model?.model ??
            "TRACE-X V1"
          }
          detail="Frozen production model"
        />

        <RuleCard
          title="THRESHOLD"
          value={
            model?.threshold ??
            0.76
          }
          detail="FLAG decision threshold"
        />

        <RuleCard
          title="FEATURES"
          value={
            model?.feature_count ??
            38
          }
          detail="Exact model feature contract"
        />

        <RuleCard
          title="BOOSTING ROUNDS"
          value={
            model?.boosting_rounds ??
            800
          }
          detail="Frozen XGBoost rounds"
        />

        <RuleCard
          title="DATASET"
          value={formatNumber(
            summary.total_transactions
          )}
          detail="Authoritative transactions"
        />

        <RuleCard
          title="FLAG RATE"
          value={formatPercent(
            summary.laundering_rate
          )}
          detail="Observed dataset rate"
        />

      </div>

    </div>
  );
}


function RuleCard({
  title,
  value,
  detail,
}) {
  return (
    <div className="rule-card">

      <div className="rule-level">
        <span />
        ACTIVE
      </div>

      <span>
        {title}
      </span>

      <strong>
        {value}
      </strong>

      <p>
        {detail}
      </p>

    </div>
  );
}


/* ============================================================
   AI
   ============================================================ */

function AIPage({
  prediction,
  aiInvestigation,
  onAnalyze,
  onRunAI,
}) {
  const ai = aiInvestigation?.ai;
  const trace = aiInvestigation?.trace_x;
  const historical =
    aiInvestigation?.historical_features;

  return (
    <div className="page-content">

      <section className="ai-hero">

        <div className="ai-orb">
          <BrainCircuit size={40} />
        </div>

        <div>

          <div className="eyebrow">
            <Sparkles size={12} />
            TRACE-X INTELLIGENCE
          </div>

          <h2>
            AI Investigation
          </h2>

          <p>
            TRACE-X V1 makes the authoritative
            risk decision. Gemini converts the
            supplied TRACE-X evidence into an
            investigator-ready briefing.
          </p>

          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              marginTop: 12,
              padding: "7px 12px",
              borderRadius: 999,
              border:
                "1px solid rgba(34,211,238,.25)",
              background:
                "rgba(34,211,238,.06)",
              color: "#67e8f9",
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.08em",
            }}
          >
            <span>
              ● GROUNDED IN TRACE-X EVIDENCE
            </span>
          </div>

          <div
            style={{
              display: "flex",
              gap: 10,
              marginTop: 18,
              flexWrap: "wrap",
            }}
          >
            <button
              type="button"
              className="primary-button"
              onClick={onAnalyze}
            >
              <Zap size={17} />
              Analyze Transaction
            </button>

            <button
              type="button"
              className="primary-button"
              onClick={onRunAI}
              disabled={
                aiInvestigation?.loading
              }
            >
              {aiInvestigation?.loading ? (
                <>
                  <RefreshCw
                    size={17}
                    className="spin"
                  />
                  Generating Investigation...
                </>
              ) : (
                <>
                  <BrainCircuit size={17} />
                  Generate AI Investigation
                </>
              )}
            </button>
          </div>

          <div
            style={{
              marginTop: 10,
              color: "#64748b",
              fontSize: 11,
            }}
          >
            TRACE-X V1 decides. Gemini explains.
          </div>

        </div>
      </section>


      {prediction &&
        !prediction.loading &&
        !prediction.error && (

        <div className="prediction-card">

          <div>
            <span>RISK SCORE</span>

            <strong>
              {(
                Number(
                  prediction.risk_score ?? 0
                ) * 100
              ).toFixed(4)}
              %
            </strong>
          </div>

          <div>
            <span>THRESHOLD</span>

            <strong>
              {prediction.threshold ?? "—"}
            </strong>
          </div>

          <div>
            <span>DECISION</span>

            <strong
              className={
                prediction.decision ===
                "FLAG"
                  ? "danger-text"
                  : "success-text"
              }
            >
              {prediction.decision ?? "—"}
            </strong>
          </div>

        </div>
      )}


      {aiInvestigation?.error && (
        <div
          className="error-box"
          style={{
            marginTop: 16,
            marginBottom: 16,
          }}
        >
          <strong>
            AI analysis unavailable
          </strong>

          <br />

          TRACE-X V1 remains operational.

          <br />

          {aiInvestigation.error}
        </div>
      )}


      {aiInvestigation?.loading && (
        <section
          className="panel"
          style={{
            marginTop: 18,
            padding: 24,
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              color: "#94a3b8",
            }}
          >
            <RefreshCw
              size={18}
              className="spin"
            />

            <span>
              TRACE-X evidence is being prepared
              for the AI investigation...
            </span>
          </div>
        </section>
      )}


      {ai &&
        !aiInvestigation?.loading &&
        !aiInvestigation?.error && (

        <>
          <section
            className="panel"
            style={{
              marginTop: 18,
            }}
          >
            <PanelHeader
              icon={
                <Sparkles size={17} />
              }
              title="AI Investigation Brief"
              subtitle="Grounded explanation of TRACE-X evidence"
            />

            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(3, minmax(0, 1fr))",
                gap: 14,
                padding: 18,
              }}
            >

              <div className="evidence">
                <span>
                  TRACE-X RISK SCORE
                </span>

                <strong
                  className="danger-text"
                >
                  {(
                    Number(
                      trace?.risk_score ??
                        prediction?.risk_score ??
                        0
                    ) * 100
                  ).toFixed(4)}
                  %
                </strong>
              </div>

              <div className="evidence">
                <span>
                  THRESHOLD
                </span>

                <strong>
                  {trace?.threshold ??
                    prediction?.threshold ??
                    "—"}
                </strong>
              </div>

              <div className="evidence">
                <span>
                  TRACE-X DECISION
                </span>

                <strong
                  className={
                    (
                      trace?.decision ??
                      prediction?.decision
                    ) === "FLAG"
                      ? "danger-text"
                      : "success-text"
                  }
                >
                  {trace?.decision ??
                    prediction?.decision ??
                    "—"}
                </strong>
              </div>

            </div>

            <div
              style={{
                margin: "0 18px 18px",
                padding: 18,
                borderRadius: 14,
                background:
                  "rgba(15,23,42,.55)",
                border:
                  "1px solid rgba(148,163,184,.08)",
                color: "#cbd5e1",
                lineHeight: 1.7,
              }}
            >
              <div
                style={{
                  marginBottom: 8,
                  color: "#67e8f9",
                  fontWeight: 700,
                  fontSize: 12,
                  letterSpacing:
                    "0.08em",
                  textTransform:
                    "uppercase",
                }}
              >
                Investigation Summary
              </div>

              {ai.summary}
            </div>

          </section>


          <div
            className="dashboard-grid"
            style={{
              marginTop: 18,
            }}
          >

            <section className="panel">

              <PanelHeader
                icon={
                  <AlertTriangle
                    size={17}
                  />
                }
                title="Why Was It Flagged?"
                subtitle="Evidence-grounded reasons"
              />

              <div
                style={{
                  padding: 18,
                }}
              >
                {(
                  ai.why_flagged || []
                ).map(
                  (item, index) => (
                    <div
                      key={index}
                      style={{
                        display: "flex",
                        gap: 10,
                        padding:
                          "11px 0",
                        borderBottom:
                          "1px solid rgba(148,163,184,.08)",
                        color: "#cbd5e1",
                        lineHeight: 1.6,
                      }}
                    >
                      <span
                        style={{
                          color: "#fb7185",
                          fontWeight: 800,
                        }}
                      >
                        •
                      </span>

                      <span>
                        {item}
                      </span>
                    </div>
                  )
                )}
              </div>

            </section>


            <section className="panel">

              <PanelHeader
                icon={
                  <Target size={17} />
                }
                title="Strongest Evidence"
                subtitle="Traceable evidence sources"
              />

              <div
                style={{
                  padding: 18,
                }}
              >
                {(
                  ai.strongest_evidence ||
                  []
                ).map(
                  (item, index) => (
                    <div
                      key={index}
                      style={{
                        marginBottom: 12,
                        padding: 13,
                        borderRadius: 12,
                        background:
                          "rgba(15,23,42,.55)",
                        border:
                          "1px solid rgba(148,163,184,.08)",
                      }}
                    >
                      <strong
                        style={{
                          display: "block",
                          color: "#e2e8f0",
                          marginBottom: 5,
                        }}
                      >
                        {item.signal}
                      </strong>

                      <div
                        style={{
                          color: "#67e8f9",
                          marginBottom: 4,
                        }}
                      >
                        {item.value}
                      </div>

                      <small
                        style={{
                          color: "#64748b",
                        }}
                      >
                        Source:{" "}
                        {item.source}
                      </small>
                    </div>
                  )
                )}
              </div>

            </section>


            <section className="panel">

              <PanelHeader
                icon={
                  <CheckCircle2
                    size={17}
                  />
                }
                title="Recommended Actions"
                subtitle="Investigator next steps"
              />

              <div
                style={{
                  padding: 18,
                }}
              >
                {(
                  ai.recommended_actions ||
                  []
                ).map(
                  (item, index) => (
                    <div
                      key={index}
                      style={{
                        display: "flex",
                        gap: 12,
                        padding: "9px 0",
                        color: "#cbd5e1",
                        lineHeight: 1.6,
                      }}
                    >
                      <strong
                        style={{
                          color: "#22d3ee",
                        }}
                      >
                        {index + 1}.
                      </strong>

                      <span>
                        {item}
                      </span>
                    </div>
                  )
                )}
              </div>

            </section>


            <section className="panel">

              <PanelHeader
                icon={
                  <Search size={17} />
                }
                title="Follow-up Questions"
                subtitle="Suggested investigator questions"
              />

              <div
                style={{
                  padding: 18,
                }}
              >
                {(
                  ai.follow_up_questions ||
                  []
                ).map(
                  (item, index) => (
                    <div
                      key={index}
                      style={{
                        display: "inline-block",
                        margin:
                          "5px 7px 5px 0",
                        padding:
                          "9px 12px",
                        borderRadius: 10,
                        border:
                          "1px solid rgba(34,211,238,.18)",
                        background:
                          "rgba(34,211,238,.04)",
                        color: "#a5f3fc",
                        fontSize: 12,
                      }}
                    >
                      {item}
                    </div>
                  )
                )}
              </div>

            </section>

          </div>


          {historical && (
            <section
              className="panel"
              style={{
                marginTop: 18,
              }}
            >

              <PanelHeader
                icon={
                  <Database size={17} />
                }
                title="TRACE-X Historical Evidence"
                subtitle="Raw evidence supplied to the AI layer"
              />

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns:
                    "repeat(3, minmax(0, 1fr))",
                  gap: 12,
                  padding: 18,
                }}
              >
                {Object.entries(
                  historical
                ).map(
                  ([key, value]) => (
                    <div
                      key={key}
                      className="evidence"
                    >
                      <span>
                        {key.replaceAll(
                          "_",
                          " "
                        )}
                      </span>

                      <strong>
                        {typeof value ===
                        "number"
                          ? value.toLocaleString()
                          : String(value)}
                      </strong>
                    </div>
                  )
                )}
              </div>

            </section>
          )}


          <section
            className="panel"
            style={{
              marginTop: 18,
            }}
          >
            <div
              style={{
                padding: 16,
                textAlign: "center",
                color: "#64748b",
                fontSize: 11,
                lineHeight: 1.6,
              }}
            >
              {ai.disclaimer ||
                "TRACE-X V1 made the risk decision. Gemini generated the investigation explanation from supplied evidence."}
            </div>
          </section>
        </>
      )}

    </div>
  );
}


/* ============================================================
   CASE DRAWER
   ============================================================ */

function CaseDrawer({
  transaction,
  onClose,
  onAnalyze,
}) {
  const fields = [
    [
      "Timestamp",
      transaction.timestamp,
    ],
    [
      "From Bank",
      transaction.from_bank,
    ],
    [
      "Sender",
      transaction.sender_account,
    ],
    [
      "To Bank",
      transaction.to_bank,
    ],
    [
      "Receiver",
      transaction.receiver_account,
    ],
    [
      "Amount Received",
      formatMoney(
        transaction.amount_received
      ),
    ],
    [
      "Amount Paid",
      formatMoney(
        transaction.amount_paid
      ),
    ],
    [
      "Payment Format",
      transaction.payment_format,
    ],
  ];

  return (
    <div
      className="drawer-backdrop"
      onMouseDown={(e) => {
        if (
          e.target ===
          e.currentTarget
        ) {
          onClose();
        }
      }}
    >

      <aside className="investigation-drawer">

        <div className="drawer-header">

          <div>
            <div className="eyebrow">
              CASE INVESTIGATION
            </div>

            <h2>
              Transaction Evidence
            </h2>
          </div>

          <button
            className="icon-button"
            type="button"
            onClick={
              onClose
            }
          >
            <X size={18} />
          </button>

        </div>


        <div className="case-risk">

          <AlertTriangle
            size={22}
          />

          <div>
            <span>
              DATASET STATUS
            </span>

            <strong>
              FLAG
            </strong>
          </div>

        </div>


        <div className="evidence-grid">

          {fields.map(
            ([label, value]) => (
              <Evidence
                key={label}
                label={label}
                value={value}
              />
            )
          )}

        </div>


        <button
          className="primary-button full-button"
          type="button"
          onClick={
            onAnalyze
          }
        >
          <BrainCircuit
            size={17}
          />
          Run TRACE-X Analysis
        </button>

      </aside>

    </div>
  );
}


function Evidence({
  label,
  value,
}) {
  return (
    <div className="evidence">

      <span>
        {label}
      </span>

      <strong>
        {value ?? "—"}
      </strong>

    </div>
  );
}


/* ============================================================
   ANALYZER
   ============================================================ */

function Analyzer({
  form,
  setForm,
  prediction,
  onAnalyze,
  onClose,
}) {
  function update(
    field,
    value
  ) {
    setForm(
      (current) => ({
        ...current,
        [field]: value,
      })
    );
  }

  const fields = [
    [
      "timestamp",
      "Timestamp",
      "text",
    ],
    [
      "from_bank",
      "From Bank",
      "text",
    ],
    [
      "sender_account",
      "Sender Account",
      "text",
    ],
    [
      "to_bank",
      "To Bank",
      "text",
    ],
    [
      "receiver_account",
      "Receiver Account",
      "text",
    ],
    [
      "amount_received",
      "Amount Received",
      "number",
    ],
    [
      "receiving_currency",
      "Receiving Currency",
      "text",
    ],
    [
      "amount_paid",
      "Amount Paid",
      "number",
    ],
    [
      "payment_currency",
      "Payment Currency",
      "text",
    ],
    [
      "payment_format",
      "Payment Format",
      "text",
    ],
  ];

  return (
    <div
      className="drawer-backdrop"
      onMouseDown={(e) => {
        if (
          e.target ===
          e.currentTarget
        ) {
          onClose();
        }
      }}
    >

      <div className="analyzer-modal">

        <div className="drawer-header">

          <div>

            <div className="eyebrow">
              TRACE-X V1
            </div>

            <h2>
              Transaction Analyzer
            </h2>

          </div>

          <button
            className="icon-button"
            type="button"
            onClick={
              onClose
            }
          >
            <X size={18} />
          </button>

        </div>


        <div className="form-grid">

          {fields.map(
            ([field, label, type]) => (
              <Field
                key={field}
                label={label}
                type={type}
                value={
                  form[field]
                }
                onChange={(value) =>
                  update(
                    field,
                    value
                  )
                }
              />
            )
          )}

        </div>


        <button
          className="primary-button full-button"
          type="button"
          disabled={
            prediction?.loading
          }
          onClick={
            onAnalyze
          }
        >

          {prediction?.loading ? (
            <>
              <RefreshCw
                size={17}
                className="spin"
              />

              Running TRACE-X...
            </>
          ) : (
            <>
              <Zap size={17} />

              Run TRACE-X V1
            </>
          )}

        </button>


        {prediction?.error && (
          <div className="error-box">
            {prediction.error}
          </div>
        )}


        {prediction &&
          !prediction.loading &&
          !prediction.error && (

            <div className="analysis-result">

              <div className="result-score">

                <span>
                  RISK SCORE
                </span>

                <strong>
                  {(
                    Number(
                      prediction.risk_score ??
                        0
                    ) * 100
                  ).toFixed(5)}
                  %
                </strong>

              </div>


              <div className="result-decision">

                <span>
                  DECISION
                </span>

                <strong
                  className={
                    prediction.decision ===
                    "FLAG"
                      ? "danger-text"
                      : "success-text"
                  }
                >
                  {prediction.decision}
                </strong>

              </div>


              {prediction.historical_features && (
                <div className="history-box">

                  <h4>
                    Historical Intelligence
                  </h4>

                  {Object.entries(
                    prediction.historical_features
                  ).map(
                    ([key, value]) => (
                      <div
                        className="history-row"
                        key={key}
                      >

                        <span>
                          {key}
                        </span>

                        <strong>
                          {typeof value ===
                          "number"
                            ? value.toLocaleString()
                            : String(value)}
                        </strong>

                      </div>
                    )
                  )}

                </div>
              )}

            </div>
          )}

      </div>

    </div>
  );
}


/* ============================================================
   FIELD
   ============================================================ */

function Field({
  label,
  value,
  type = "text",
  onChange,
}) {
  return (
    <label className="input-group">

      <span>
        {label}
      </span>

      <input
        type={type}
        value={value ?? ""}
        onChange={(event) => {
          const value =
            type === "number"
              ? event.target.value === ""
                ? 0
                : Number(
                    event.target.value
                  )
              : event.target.value;

          onChange(value);
        }}
      />

    </label>
  );
}