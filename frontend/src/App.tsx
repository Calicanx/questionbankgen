import { useState, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';

// Import mhchem extension for chemistry formulas (\ce, \pu)
import 'katex/contrib/mhchem/mhchem.js';

const API_BASE = 'http://localhost:8001/api';

// Convert web+graphie:// URLs to HTTPS (Khan Academy custom protocol)
const convertImageUrl = (url: string | undefined): string => {
  if (!url) return '';
  if (url.startsWith('web+graphie://')) {
    return url.replace('web+graphie://', 'https://') + '.svg';
  }
  return url;
};

interface Hint {
  content: string;
  widgets?: Record<string, any>;
  images?: Record<string, any>;
  replace?: boolean;
}

interface SolutionStep {
  step: number;
  content: string;
  action: string;
}

interface Solution {
  steps: SolutionStep[];
  final_answer: string;
}

interface Question {
  _id: string;
  question: {
    content: string;
    widgets: Record<string, any>;
    images: Record<string, any>;
  };
  hints: Hint[];
  solution?: Solution;
}

interface GeneratedResult {
  id: string;
  source: Question;
  generated: Question;
  metadata: any;
}

// Light Mode Styles
const styles = {
  container: {
    minHeight: '100vh',
    backgroundColor: '#f8fafc',
    color: '#1e293b',
    fontFamily: 'system-ui, -apple-system, sans-serif',
  } as React.CSSProperties,
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 32px',
    borderBottom: '1px solid #e2e8f0',
    backgroundColor: '#ffffff',
    boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
  } as React.CSSProperties,
  title: {
    fontSize: '20px',
    fontWeight: 700,
    color: '#0f172a',
  } as React.CSSProperties,
  subtitle: {
    fontSize: '12px',
    color: '#64748b',
    marginTop: '2px',
  } as React.CSSProperties,
  nav: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  } as React.CSSProperties,
  navButton: {
    padding: '8px 16px',
    backgroundColor: '#ffffff',
    border: '1px solid #e2e8f0',
    borderRadius: '8px',
    color: '#475569',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 500,
    transition: 'all 0.15s',
  } as React.CSSProperties,
  navButtonDisabled: {
    opacity: 0.4,
    cursor: 'not-allowed',
  } as React.CSSProperties,
  counter: {
    fontSize: '14px',
    color: '#64748b',
    minWidth: '80px',
    textAlign: 'center' as const,
    fontWeight: 500,
  } as React.CSSProperties,
  main: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '24px',
    padding: '24px 32px',
    maxWidth: '1600px',
    margin: '0 auto',
  } as React.CSSProperties,
  card: {
    backgroundColor: '#ffffff',
    borderRadius: '16px',
    border: '1px solid #e2e8f0',
    overflow: 'hidden',
    boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
  } as React.CSSProperties,
  cardHeader: {
    padding: '16px 24px',
    borderBottom: '1px solid #e2e8f0',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#f8fafc',
  } as React.CSSProperties,
  cardTitleOriginal: {
    fontSize: '13px',
    fontWeight: 700,
    color: '#2563eb',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
  } as React.CSSProperties,
  cardTitleGenerated: {
    fontSize: '13px',
    fontWeight: 700,
    color: '#7c3aed',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
  } as React.CSSProperties,
  cardId: {
    fontSize: '11px',
    color: '#94a3b8',
    fontFamily: 'monospace',
    backgroundColor: '#f1f5f9',
    padding: '4px 8px',
    borderRadius: '4px',
  } as React.CSSProperties,
  cardContent: {
    padding: '24px',
    minHeight: '400px',
  } as React.CSSProperties,
  questionText: {
    fontSize: '16px',
    lineHeight: 1.8,
    color: '#334155',
    marginBottom: '20px',
  } as React.CSSProperties,
  image: {
    maxWidth: '100%',
    borderRadius: '12px',
    marginBottom: '16px',
    border: '1px solid #e2e8f0',
    boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
  } as React.CSSProperties,
  choicesList: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '12px',
    marginTop: '20px',
  } as React.CSSProperties,
  choice: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '14px',
    padding: '16px 18px',
    backgroundColor: '#f8fafc',
    borderRadius: '12px',
    border: '1px solid #e2e8f0',
    transition: 'all 0.15s',
  } as React.CSSProperties,
  choiceCorrect: {
    borderColor: '#22c55e',
    backgroundColor: '#f0fdf4',
  } as React.CSSProperties,
  choiceRadio: {
    width: '20px',
    height: '20px',
    borderRadius: '50%',
    border: '2px solid #cbd5e1',
    flexShrink: 0,
    marginTop: '2px',
  } as React.CSSProperties,
  choiceRadioCorrect: {
    borderColor: '#22c55e',
    backgroundColor: '#22c55e',
  } as React.CSSProperties,
  loading: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '300px',
    color: '#94a3b8',
    fontSize: '15px',
  } as React.CSSProperties,
  badge: {
    display: 'inline-block',
    padding: '4px 10px',
    backgroundColor: '#f1f5f9',
    borderRadius: '6px',
    fontSize: '11px',
    color: '#64748b',
    marginLeft: '10px',
    fontWeight: 500,
  } as React.CSSProperties,
  sourceLabel: {
    fontSize: '11px',
    color: '#64748b',
    marginTop: '8px',
  } as React.CSSProperties,
  hintsSection: {
    marginTop: '24px',
    paddingTop: '20px',
    borderTop: '1px solid #e2e8f0',
  } as React.CSSProperties,
  hintsHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    cursor: 'pointer',
    padding: '12px 16px',
    backgroundColor: '#f8fafc',
    borderRadius: '8px',
    border: '1px solid #e2e8f0',
    marginBottom: '12px',
  } as React.CSSProperties,
  hintsTitle: {
    fontSize: '14px',
    fontWeight: 600,
    color: '#475569',
  } as React.CSSProperties,
  hintsBadge: {
    fontSize: '12px',
    color: '#64748b',
    backgroundColor: '#e2e8f0',
    padding: '2px 8px',
    borderRadius: '12px',
  } as React.CSSProperties,
  hintItem: {
    padding: '16px',
    backgroundColor: '#fffbeb',
    borderRadius: '8px',
    border: '1px solid #fde68a',
    marginBottom: '12px',
  } as React.CSSProperties,
  hintNumber: {
    fontSize: '12px',
    fontWeight: 600,
    color: '#d97706',
    marginBottom: '8px',
  } as React.CSSProperties,
  hintContent: {
    fontSize: '14px',
    lineHeight: 1.7,
    color: '#78350f',
  } as React.CSSProperties,
  solutionSection: {
    marginTop: '24px',
    paddingTop: '20px',
    borderTop: '2px solid #7c3aed',
  } as React.CSSProperties,
  solutionHeader: {
    fontSize: '16px',
    fontWeight: 700,
    color: '#7c3aed',
    marginBottom: '16px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  } as React.CSSProperties,
  stepItem: {
    display: 'flex',
    gap: '16px',
    padding: '16px',
    backgroundColor: '#faf5ff',
    borderRadius: '8px',
    border: '1px solid #e9d5ff',
    marginBottom: '12px',
  } as React.CSSProperties,
  stepNumber: {
    width: '32px',
    height: '32px',
    borderRadius: '50%',
    backgroundColor: '#7c3aed',
    color: '#fff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '14px',
    fontWeight: 600,
    flexShrink: 0,
  } as React.CSSProperties,
  stepContent: {
    flex: 1,
  } as React.CSSProperties,
  stepAction: {
    fontSize: '12px',
    color: '#6b21a8',
    fontWeight: 500,
    marginBottom: '4px',
  } as React.CSSProperties,
  stepMath: {
    fontSize: '16px',
    color: '#581c87',
    lineHeight: 1.6,
  } as React.CSSProperties,
  finalAnswer: {
    padding: '16px 20px',
    backgroundColor: '#dcfce7',
    borderRadius: '8px',
    border: '2px solid #22c55e',
    marginTop: '16px',
  } as React.CSSProperties,
  finalAnswerLabel: {
    fontSize: '12px',
    fontWeight: 600,
    color: '#166534',
    marginBottom: '4px',
  } as React.CSSProperties,
  finalAnswerValue: {
    fontSize: '18px',
    fontWeight: 600,
    color: '#15803d',
  } as React.CSSProperties,
};

const QuestionRenderer = ({ question, widgets }: { question: string; widgets: Record<string, any> }) => {
  if (!question) return <div style={{ color: '#94a3b8' }}>No content</div>;

  const parts = question.split(/(\[\[☃ [^\]]+\]\])/g);

  const renderWidget = (widgetId: string) => {
    const w = widgets[widgetId];
    if (!w) return null;

    if (w.type === 'image' && w.options?.backgroundImage?.url) {
      const imageUrl = convertImageUrl(w.options.backgroundImage.url);
      return (
        <div key={widgetId} style={{ margin: '20px 0' }}>
          <img
            src={imageUrl}
            alt={w.options.alt || "Question image"}
            style={styles.image}
            onError={(e) => {
              // Fallback: try without .svg extension if SVG fails
              const target = e.target as HTMLImageElement;
              if (target.src.endsWith('.svg')) {
                target.src = target.src.slice(0, -4);
              }
            }}
          />
        </div>
      );
    }

    if (w.type === 'numeric-input') {
      return (
        <input
          key={widgetId}
          type="text"
          placeholder="?"
          style={{
            width: '80px',
            padding: '8px 12px',
            backgroundColor: '#ffffff',
            border: '2px solid #e2e8f0',
            borderRadius: '8px',
            color: '#1e293b',
            margin: '0 6px',
            fontSize: '16px',
            textAlign: 'center' as const,
          }}
        />
      );
    }

    return null;
  };

  const radioWidgets = Object.entries(widgets).filter(([, w]) => w.type === 'radio');

  return (
    <div>
      <div style={styles.questionText}>
        {parts.map((part, i) => {
          if (part.startsWith('[[☃')) {
            const match = part.match(/\[\[☃ ([^\]]+)\]\]/);
            if (match) {
              const widgetId = match[1];
              const w = widgets[widgetId];
              if (w?.type === 'radio') return null;
              return renderWidget(widgetId);
            }
          }
          return (
            <ReactMarkdown
              key={i}
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeRaw, rehypeKatex]}
              components={{
                p: ({ children }) => <span>{children}</span>,
              }}
            >
              {part}
            </ReactMarkdown>
          );
        })}
      </div>

      {radioWidgets.map(([id, w]) => (
        <div key={id} style={styles.choicesList}>
          {w.options?.choices?.map((choice: any, ci: number) => (
            <div
              key={ci}
              style={styles.choice}
            >
              <div style={styles.choiceRadio} />
              <div style={{ flex: 1, color: '#334155' }}>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeRaw, rehypeKatex]}
                >
                  {choice.content}
                </ReactMarkdown>
              </div>
{/* Correct answer indicator hidden for clean comparison */}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

// Process hint content to fix LaTeX rendering issues
const processHintContent = (content: string): string => {
  if (!content) return '';

  let processed = content;

  // Khan Academy hints have complex nested align structures like:
  // $\begin{align}\begin{align}...\end{align}&\begin{align}...\end{align}\end{align}$
  // Strategy: Convert outer delimiters and remove all inner align tags

  // Step 1: Convert $\begin{align} to $$\begin{aligned} with newlines for display math
  // Note: In JS replace(), $$ means literal $, so $$$$ = $$
  // Add newlines so remarkMath detects it as display math block
  processed = processed.replace(/\$\\begin\{align\*?\}/g, '\n\n$$$$\\begin{aligned}');

  // Step 2: Convert \end{align}$ to \end{aligned}$$ with newlines
  processed = processed.replace(/\\end\{align\*?\}\$/g, '\\end{aligned}$$$$\n\n');

  // Step 3: Remove ALL remaining \begin{align} tags (nested ones)
  processed = processed.replace(/\\begin\{align\*?\}/g, '');

  // Step 4: Remove ALL remaining \end{align} tags (nested ones)
  processed = processed.replace(/\\end\{align\*?\}/g, '');

  // Step 5: Clean up any doubled $$ or empty aligned blocks
  // Match 4 consecutive $ and replace with 2
  processed = processed.replace(/\$\$\$\$/g, '$$$$');
  processed = processed.replace(/\\begin\{aligned\}\s*\\begin\{aligned\}/g, '\\begin{aligned}');
  processed = processed.replace(/\\end\{aligned\}\s*\\end\{aligned\}/g, '\\end{aligned}');

  return processed;
};

// Hints Section Component
const HintsSection = ({ hints }: { hints: Hint[] }) => {
  const [expanded, setExpanded] = useState(false);

  if (!hints || hints.length === 0) return null;

  return (
    <div style={styles.hintsSection}>
      <div style={styles.hintsHeader} onClick={() => setExpanded(!expanded)}>
        <span style={{ fontSize: '16px' }}>{expanded ? '▼' : '▶'}</span>
        <span style={styles.hintsTitle}>Hints</span>
        <span style={styles.hintsBadge}>{hints.length}</span>
      </div>
      {expanded && (
        <div>
          {hints.map((hint, i) => (
            <div key={i} style={styles.hintItem}>
              <div style={styles.hintNumber}>Hint {i + 1}</div>
              <div style={styles.hintContent}>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeRaw, rehypeKatex]}
                >
                  {processHintContent(hint.content)}
                </ReactMarkdown>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Solution Steps Component (for generated questions only)
const SolutionSteps = ({ solution }: { solution?: Solution }) => {
  if (!solution || !solution.steps || solution.steps.length === 0) return null;

  return (
    <div style={styles.solutionSection}>
      <div style={styles.solutionHeader}>
        <span>📝</span>
        <span>Step-by-Step Solution</span>
      </div>
      {solution.steps.map((step, i) => (
        <div key={i} style={styles.stepItem}>
          <div style={styles.stepNumber}>{step.step}</div>
          <div style={styles.stepContent}>
            <div style={styles.stepAction}>{step.action}</div>
            <div style={styles.stepMath}>
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeRaw, rehypeKatex]}
              >
                {step.content}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      ))}
      {solution.final_answer && (
        <div style={styles.finalAnswer}>
          <div style={styles.finalAnswerLabel}>Final Answer</div>
          <div style={styles.finalAnswerValue}>
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeRaw, rehypeKatex]}
            >
              {solution.final_answer}
            </ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
};

function App() {
  const [results, setResults] = useState<GeneratedResult[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);

  const currentResult = results[currentIndex];

  useEffect(() => {
    fetchResults();
  }, []);

  const fetchResults = async () => {
    try {
      setLoading(true);
      const res = await axios.get(`${API_BASE}/generated`);
      setResults(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handlePrev = () => {
    if (currentIndex > 0) setCurrentIndex(prev => prev - 1);
  };

  const handleNext = () => {
    if (currentIndex < results.length - 1) setCurrentIndex(prev => prev + 1);
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div>
          <div style={styles.title}>Question Comparison</div>
          <div style={styles.subtitle}>Original (MongoDB) vs AI Generated (Gemini)</div>
        </div>
        <div style={styles.nav}>
          <button
            style={{
              ...styles.navButton,
              ...(currentIndex === 0 ? styles.navButtonDisabled : {}),
            }}
            onClick={handlePrev}
            disabled={currentIndex === 0}
          >
            ← Prev
          </button>
          <span style={styles.counter}>
            {results.length > 0 ? `${currentIndex + 1} / ${results.length}` : '...'}
          </span>
          <button
            style={{
              ...styles.navButton,
              ...(currentIndex >= results.length - 1 ? styles.navButtonDisabled : {}),
            }}
            onClick={handleNext}
            disabled={currentIndex >= results.length - 1}
          >
            Next →
          </button>
          <button
            style={{...styles.navButton, backgroundColor: '#2563eb', color: '#fff', border: 'none'}}
            onClick={fetchResults}
          >
            Refresh
          </button>
        </div>
      </header>

      <main style={styles.main}>
        {/* Original Question from MongoDB */}
        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <div>
              <span style={styles.cardTitleOriginal}>Original Question</span>
              <div style={styles.sourceLabel}>From: scraped_questions collection</div>
            </div>
            <span style={styles.cardId}>{currentResult?.source?._id?.slice(-8) || '...'}</span>
          </div>
          <div style={styles.cardContent}>
            {loading ? (
              <div style={styles.loading}>Loading questions...</div>
            ) : currentResult?.source ? (
              <>
                <QuestionRenderer
                  question={currentResult.source.question?.content || ''}
                  widgets={currentResult.source.question?.widgets || {}}
                />
                <HintsSection hints={currentResult.source.hints || []} />
              </>
            ) : (
              <div style={styles.loading}>No data available</div>
            )}
          </div>
        </div>

        {/* AI Generated Question */}
        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <div>
              <span style={styles.cardTitleGenerated}>AI Generated</span>
              <span style={styles.badge}>{currentResult?.metadata?.llm_model || 'gemini-2.0-flash'}</span>
            </div>
            <span style={styles.cardId}>GEN_{currentIndex}</span>
          </div>
          <div style={styles.cardContent}>
            {loading ? (
              <div style={styles.loading}>Loading questions...</div>
            ) : currentResult?.generated ? (
              <>
                <QuestionRenderer
                  question={currentResult.generated.question?.content || ''}
                  widgets={currentResult.generated.question?.widgets || {}}
                />
                <HintsSection hints={currentResult.generated.hints || []} />
                <SolutionSteps solution={currentResult.generated.solution} />
              </>
            ) : (
              <div style={styles.loading}>No data available</div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
