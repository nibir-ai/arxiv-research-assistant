"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { searchPapers, healthCheck, type SearchResult } from "@/lib/api";

// ── Icons (inline SVG for zero-dep speed) ──────────────────────────────────

function IconSearch({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="8.5" cy="8.5" r="5.5" />
      <path d="M13 13l4 4" />
    </svg>
  );
}

function IconArrowRight({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 10h12m-5-5l5 5-5 5" />
    </svg>
  );
}

function IconExternalLink({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 2h8v8M14 2L6 10" />
    </svg>
  );
}

function IconPaper({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 2h8l4 4v10a2 2 0 01-2 2H6a2 2 0 01-2-2V4a2 2 0 012-2z" />
      <path d="M14 2v4h4M8 10h4M8 14h6" />
    </svg>
  );
}

// Sparkle component styled to accept className
function IconSparkle({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor">
      <path d="M10 1l1.753 5.247L17 8l-5.247 1.753L10 15l-1.753-5.247L3 8l5.247-1.753z" />
    </svg>
  );
}

function IconClock({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
      <circle cx="8" cy="8" r="6" />
      <path d="M8 4.5V8l2.5 1.5" />
    </svg>
  );
}

function IconTune({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 14h4M3 6h10M13 14h4M7 6h10M5 3v6m8 8v-6" />
    </svg>
  );
}

// ── Status indicator ───────────────────────────────────────────────────────

function StatusIndicator({ online }: { online: boolean | null }) {
  const dotClass =
    online === null
      ? "status-dot status-dot--checking"
      : online
        ? "status-dot status-dot--online"
        : "status-dot status-dot--offline";

  const label =
    online === null ? "Checking..." : online ? "API connected" : "API offline";

  return (
    <div id="status-indicator" className="flex items-center gap-2">
      <span className={dotClass} />
      <span className="text-xs mono" style={{ color: "var(--text-quaternary)" }}>
        {label}
      </span>
    </div>
  );
}

// ── Loading state ──────────────────────────────────────────────────────────

function LoadingState() {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const intervals = [1000, 1500, 1800];
    let currentStage = 0;
    
    const nextStage = () => {
      if (currentStage < 3) {
        currentStage++;
        setStage(currentStage);
        timer = setTimeout(nextStage, intervals[currentStage - 1] || 1500);
      }
    };

    let timer = setTimeout(nextStage, 1000);
    return () => clearTimeout(timer);
  }, []);

  const stages = [
    "Encoding query & retrieving BM25 matches...",
    "Retrieving semantic matches from Qdrant vector store...",
    "Applying Cross-Encoder reranking models...",
    "Synthesizing grounded answer using Ollama & LlamaIndex..."
  ];

  return (
    <div className="fade-up" style={{ padding: "48px 0" }}>
      <div className="flex items-center justify-center" style={{ marginBottom: 32 }}>
        <div className="loader">
          <div className="loader-dots">
            <div className="loader-dot" />
            <div className="loader-dot" />
            <div className="loader-dot" />
          </div>
          <span className="mono">{stages[stage]}</span>
        </div>
      </div>
      {/* Skeleton preview */}
      <div className="answer-panel" style={{ opacity: 0.5 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div className="shimmer-line" style={{ width: "90%" }} />
          <div className="shimmer-line" style={{ width: "75%" }} />
          <div className="shimmer-line" style={{ width: "82%" }} />
          <div className="shimmer-line" style={{ width: "60%" }} />
        </div>
      </div>
    </div>
  );
}

// ── Citation card ──────────────────────────────────────────────────────────

function CitationCard({
  citation,
  index,
}: {
  citation: SearchResult["citations"][0];
  index: number;
}) {
  const authorsText = Array.isArray(citation.authors)
    ? citation.authors.slice(0, 3).join(", ") +
      (citation.authors.length > 3 ? " et al." : "")
    : citation.authors;

  return (
    <a
      href={citation.url}
      target="_blank"
      rel="noopener noreferrer"
      className={`citation-card fade-up stagger-${Math.min(index, 5)}`}
      id={`citation-${index}`}
    >
      <span className="citation-index">{index}</span>
      <div style={{ minWidth: 0, flex: 1 }}>
        <p className="citation-title">{citation.title}</p>
        <p className="citation-authors">{authorsText}</p>
        {citation.published && (
          <p className="citation-date">{citation.published}</p>
        )}
      </div>
      <IconExternalLink className="citation-arrow" />
    </a>
  );
}

// ── Result panel ───────────────────────────────────────────────────────────

function ResultPanel({ result }: { result: SearchResult }) {
  // Convert [1], [2] annotations into interactive anchor links with data-ref
  const formattedAnswer = result.answer
    .replace(
      /\[(\d+)\]/g,
      '<a href="#citation-$1" class="answer-ref" data-ref="$1">[$1]</a>'
    )
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br/>");

  const handleAnswerClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement;
    if (target.classList.contains("answer-ref")) {
      e.preventDefault();
      const refId = target.getAttribute("data-ref");
      const element = document.getElementById(`citation-${refId}`);
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "center" });
        // Flash/Highlight the target card
        element.classList.add("citation-card--highlight");
        setTimeout(() => {
          element.classList.remove("citation-card--highlight");
        }, 2000);
      }
    }
  };

  return (
    <div className="fade-up" style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      {/* Answer card */}
      <div className="answer-panel">
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            marginBottom: 20,
          }}
        >
          <IconSparkle />
          <span className="label" style={{ color: "var(--accent-text-dim)" }}>
            Answer
          </span>
          <div style={{ flex: 1 }} />
          <span
            className="mono"
            style={{
              fontSize: 11,
              color: "var(--text-quaternary)",
            }}
          >
            {result.total_retrieved} papers retrieved
            {result.reranking_applied && " · reranked"}
          </span>
        </div>
        <div
          className="answer-text"
          onClick={handleAnswerClick}
          dangerouslySetInnerHTML={{
            __html: `<p>${formattedAnswer}</p>`,
          }}
        />
      </div>

      {/* Citations section */}
      {result.citations.length > 0 && (
        <div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              marginBottom: 14,
            }}
          >
            <IconPaper />
            <span className="label">Sources</span>
            <div className="divider" style={{ flex: 1 }} />
            <span
              className="mono"
              style={{ fontSize: 11, color: "var(--text-quaternary)" }}
            >
              {result.citations.length} papers
            </span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {result.citations.map((c) => (
              <CitationCard key={c.index} citation={c} index={c.index} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Example queries ────────────────────────────────────────────────────────

const EXAMPLES = [
  {
    text: "How does RAG improve LLM accuracy?",
    icon: "🔍",
  },
  {
    text: "What is LoRA and when should I use it?",
    icon: "🧬",
  },
  {
    text: "How do AI agents use tools for planning?",
    icon: "🤖",
  },
  {
    text: "Compare BM25 and dense vector retrieval",
    icon: "⚖️",
  },
  {
    text: "What makes a good LLM evaluation metric?",
    icon: "📊",
  },
  {
    text: "Explain attention mechanisms in transformers",
    icon: "🧠",
  },
];

// ── Main page ──────────────────────────────────────────────────────────────

export default function Home() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SearchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [topK, setTopK] = useState(5);
  const [useReranking, setUseReranking] = useState(true);
  const [showSettings, setShowSettings] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const settingsRef = useRef<HTMLDivElement>(null);

  // Health check
  useEffect(() => {
    healthCheck().then(setOnline);
  }, []);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Close settings popover on click outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        showSettings &&
        settingsRef.current &&
        !settingsRef.current.contains(event.target as Node) &&
        !(event.target as HTMLElement).closest("#settings-toggle")
      ) {
        setShowSettings(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showSettings]);

  const handleSearch = useCallback(
    async (q?: string) => {
      const searchQuery = (q ?? query).trim();
      if (!searchQuery || loading) return;

      setLoading(true);
      setError(null);
      setResult(null);
      setShowSettings(false);
      if (q) setQuery(q);

      try {
        const res = await searchPapers(searchQuery, topK, useReranking);
        setResult(res);
        setHistory((prev) =>
          [searchQuery, ...prev.filter((h) => h !== searchQuery)].slice(0, 5)
        );
        // Scroll to results smoothly
        setTimeout(() => {
          resultsRef.current?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
        }, 100);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Something went wrong"
        );
      } finally {
        setLoading(false);
      }
    },
    [query, loading, topK, useReranking]
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    } else if (e.key === "Escape") {
      setQuery("");
    }
  };

  const showHero = !result && !loading && !error;

  return (
    <div className="dot-grid" style={{ minHeight: "100vh", position: "relative" }}>
      {/* Ambient light effect */}
      <div className="ambient-bg" />

      {/* Header */}
      <header
        className="glass"
        style={{
          position: "sticky",
          top: 0,
          zIndex: 20,
          borderTop: "none",
          borderLeft: "none",
          borderRight: "none",
          borderRadius: 0,
        }}
      >
        <div
          style={{
            maxWidth: 860,
            margin: "0 auto",
            padding: "14px 24px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            {/* Logo */}
            <div
              style={{
                width: 34,
                height: 34,
                borderRadius: "var(--radius-sm)",
                background: "var(--accent-glow)",
                border: "1px solid rgba(139, 92, 246, 0.2)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <IconPaper />
            </div>
            <div>
              <div
                style={{
                  fontSize: 14,
                  fontWeight: 600,
                  color: "var(--text-primary)",
                  letterSpacing: "-0.01em",
                }}
              >
                ArXiv Research Assistant
              </div>
              <div
                className="mono"
                style={{
                  fontSize: 11,
                  color: "var(--text-quaternary)",
                  marginTop: 1,
                }}
              >
                Hybrid RAG · 209 papers indexed
              </div>
            </div>
          </div>
          <StatusIndicator online={online} />
        </div>
      </header>

      {/* Main content */}
      <main
        style={{
          maxWidth: 860,
          margin: "0 auto",
          padding: "0 24px",
          position: "relative",
          zIndex: 1,
        }}
      >
        {/* Hero section */}
        {showHero && (
          <div
            className="fade-up"
            style={{
              textAlign: "center",
              paddingTop: 80,
              paddingBottom: 16,
            }}
          >
            {/* Tech badge */}
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                padding: "6px 14px",
                background: "var(--accent-glow)",
                border: "1px solid rgba(139, 92, 246, 0.15)",
                borderRadius: "var(--radius-full)",
                fontSize: 12,
                color: "var(--accent-text-dim)",
                marginBottom: 28,
              }}
              className="mono"
            >
              <span
                style={{
                  width: 5,
                  height: 5,
                  borderRadius: "50%",
                  background: "var(--accent-primary)",
                  display: "inline-block",
                }}
              />
              BM25 + Dense Vectors · Cross-encoder Reranking
            </div>

            <h1
              style={{
                fontSize: "clamp(28px, 5vw, 44px)",
                fontWeight: 700,
                color: "var(--text-primary)",
                lineHeight: 1.15,
                letterSpacing: "-0.03em",
                marginBottom: 14,
              }}
            >
              Research with
              <br />
              <span
                style={{
                  background:
                    "linear-gradient(135deg, var(--accent-primary), #818cf8, var(--accent-primary))",
                  backgroundSize: "200% 100%",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  animation: "gradientShift 6s ease infinite",
                }}
              >
                grounded answers
              </span>
            </h1>
            <p
              style={{
                fontSize: 15,
                color: "var(--text-tertiary)",
                maxWidth: 480,
                margin: "0 auto",
                lineHeight: 1.7,
              }}
            >
              Ask questions about ML research and get cited answers from
              209&nbsp;indexed arXiv papers.
            </p>
          </div>
        )}

        {/* Search area */}
        <div
          className={showHero ? "fade-up stagger-2" : ""}
          style={{
            paddingTop: showHero ? 36 : 36,
            paddingBottom: 8,
          }}
        >
          <div className="search-container">
            <IconSearch />
            <input
              ref={inputRef}
              id="search-input"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about ML research..."
              className="search-input"
              disabled={loading}
              autoComplete="off"
              spellCheck={false}
            />
            
            {/* Tune parameters button absolutely positioned inside input container */}
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`settings-toggle-icon-btn ${showSettings ? "active" : ""}`}
              id="settings-toggle"
              title="Search parameters"
              type="button"
            >
              <IconTune />
            </button>

            <button
              id="search-button"
              onClick={() => handleSearch()}
              disabled={loading || !query.trim()}
              className="btn-primary"
              style={{
                position: "absolute",
                right: 6,
                top: "50%",
                transform: "translateY(-50%)",
              }}
            >
              {loading ? (
                <div className="loader-dots" style={{ gap: 3 }}>
                  <div className="loader-dot" style={{ width: 4, height: 4 }} />
                  <div className="loader-dot" style={{ width: 4, height: 4 }} />
                  <div className="loader-dot" style={{ width: 4, height: 4 }} />
                </div>
              ) : (
                <IconArrowRight />
              )}
              <span>Search</span>
            </button>

            {/* Expandable floating settings popover */}
            {showSettings && (
              <div className="settings-popover" ref={settingsRef}>
                <div className="settings-group">
                  <span className="settings-label">Top-K retrieved</span>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <input
                      type="range"
                      min="3"
                      max="10"
                      value={topK}
                      onChange={(e) => setTopK(Number(e.target.value))}
                      className="range-input"
                      id="top-k-slider"
                    />
                    <span className="settings-value">{topK}</span>
                  </div>
                </div>
                <div className="settings-group">
                  <span className="settings-label">Cross-Encoder reranking</span>
                  <label className="switch-container">
                    <input
                      type="checkbox"
                      checked={useReranking}
                      onChange={(e) => setUseReranking(e.target.checked)}
                      style={{ display: "none" }}
                      id="reranking-toggle"
                    />
                    <div className="switch-track">
                      <div className="switch-thumb" />
                    </div>
                  </label>
                </div>
              </div>
            )}
          </div>

          {/* History chips */}
          {history.length > 0 && !loading && (
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 8,
                marginTop: 14,
                alignItems: "center",
              }}
            >
              <IconClock />
              {history.map((h, i) => (
                <button
                  key={i}
                  onClick={() => handleSearch(h)}
                  className="chip"
                  id={`history-chip-${i}`}
                >
                  {h.length > 35 ? h.slice(0, 35) + "…" : h}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Example queries */}
        {showHero && history.length === 0 && (
          <div
            className="fade-up stagger-3"
            style={{ paddingTop: 20, paddingBottom: 40 }}
          >
            <p className="label" style={{ marginBottom: 14 }}>
              Try asking
            </p>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))",
                gap: 8,
              }}
            >
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  onClick={() => handleSearch(ex.text)}
                  className={`example-card fade-up stagger-${Math.min(i + 1, 5)}`}
                  id={`example-query-${i}`}
                >
                  <span style={{ fontSize: 16, lineHeight: 1, flexShrink: 0 }}>
                    {ex.icon}
                  </span>
                  <span>{ex.text}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Results area */}
        <div ref={resultsRef} style={{ scrollMarginTop: 100 }}>
          {/* Loading */}
          {loading && <LoadingState />}

          {/* Error */}
          {error && !loading && (
            <div className="error-panel fade-up" id="error-panel">
              <div style={{ fontWeight: 600, marginBottom: 4 }}>
                Search failed
              </div>
              <div style={{ opacity: 0.8 }}>{error}</div>
              {(error.includes("offline") || error.includes("fetch")) && (
                <div style={{ marginTop: 8 }}>
                  <span style={{ fontSize: 12, opacity: 0.6 }}>
                    Start the server:{" "}
                  </span>
                  <code>uvicorn app.main:app --reload --port 8000</code>
                </div>
              )}
            </div>
          )}

          {/* Result */}
          {result && !loading && <ResultPanel result={result} />}
        </div>

        {/* Spacer before footer */}
        <div style={{ height: 80 }} />
      </main>

      {/* Footer */}
      <footer style={{ borderTop: "1px solid var(--border-subtle)" }}>
        <div
          style={{
            maxWidth: 860,
            margin: "0 auto",
            padding: "16px 24px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <span
            className="mono"
            style={{ fontSize: 11, color: "var(--text-quaternary)" }}
          >
            arxiv-research-assistant
          </span>
          <span
            className="mono"
            style={{ fontSize: 11, color: "var(--text-quaternary)" }}
          >
            BM25 + Vector · Qdrant · Ollama
          </span>
        </div>
      </footer>

      {/* Gradient/Responsive keyframes and CSS rules */}
      <style>{`
        @keyframes gradientShift {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }

        /* Position the search icon inside the container */
        .search-container > svg:first-child {
          position: absolute;
          left: 20px;
          top: 50%;
          transform: translateY(-50%);
          width: 18px;
          height: 18px;
          color: var(--text-quaternary);
          pointer-events: none;
          z-index: 1;
          transition: color var(--duration-normal);
        }

        .search-container:focus-within > svg:first-child {
          color: var(--accent-primary);
        }

        /* Small icon sizing in headers */
        .answer-panel > div:first-child > svg,
        .label + svg,
        div > svg + .label {
          width: 14px;
          height: 14px;
          color: var(--accent-text-dim);
        }

        /* The sparkle icon before "Answer" */
        .answer-panel > div:first-child > svg:first-child {
          width: 15px;
          height: 15px;
          color: var(--accent-text-dim);
        }

        /* History clock icon */
        .chip ~ svg,
        div > svg:first-child {
          width: 14px;
          height: 14px;
          color: var(--text-quaternary);
          flex-shrink: 0;
        }

        /* Logo icon */
        header svg {
          width: 16px;
          height: 16px;
          color: var(--accent-text);
        }

        /* Absolute settings icon inside the input */
        .settings-toggle-icon-btn {
          position: absolute;
          right: 104px;
          top: 50%;
          transform: translateY(-50%);
          background: none;
          border: none;
          cursor: pointer;
          padding: 8px;
          border-radius: var(--radius-sm);
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--text-tertiary);
          transition: color var(--duration-fast);
          z-index: 10;
        }

        .settings-toggle-icon-btn:hover {
          color: var(--text-secondary);
        }

        .settings-toggle-icon-btn.active {
          color: var(--accent-text) !important;
        }

        /* Responsive styling overrides */
        @media (max-width: 640px) {
          .search-input {
            font-size: 14px;
            padding: 16px 84px 16px 46px !important;
          }

          .btn-primary {
            padding: 10px 16px;
            font-size: 13px;
          }

          .btn-primary > span {
            display: none;
          }

          .settings-toggle-icon-btn {
            right: 48px !important;
          }

          .answer-panel {
            padding: 20px 18px;
          }

          .settings-popover {
            width: calc(100vw - 48px);
            right: -12px;
          }
        }
      `}</style>
    </div>
  );
}