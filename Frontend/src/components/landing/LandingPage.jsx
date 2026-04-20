import { Flame, ArrowRight, Map, FileText, Zap, Shield, Activity, ChevronRight } from 'lucide-react';

const PILL_TAGS = ['XGBoost · LightGBM', 'Gemini 2.5 Flash', 'Apache Airflow', 'H3 Hexagonal Grid', 'Vertex AI', 'GCS · BigQuery'];

export default function LandingPage({ onNavigate }) {
  return (
    <div
      className="relative h-full overflow-y-auto text-white"
      style={{ background: '#000000', fontFamily: '"DM Sans", sans-serif' }}
    >
      <style>{`
        /* ── Floating orbs ── */
        @keyframes orbDrift1 {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33%       { transform: translate(60px, -40px) scale(1.08); }
          66%       { transform: translate(-30px, 50px) scale(0.94); }
        }
        @keyframes orbDrift2 {
          0%, 100% { transform: translate(0, 0) scale(1); }
          40%       { transform: translate(-70px, 30px) scale(1.12); }
          70%       { transform: translate(40px, -60px) scale(0.9); }
        }
        @keyframes orbDrift3 {
          0%, 100% { transform: translate(0, 0) scale(1); }
          50%       { transform: translate(50px, 50px) scale(1.05); }
        }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(28px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes imgFloat {
          0%, 100% { transform: perspective(1200px) rotateX(4deg) translateY(0px); }
          50%       { transform: perspective(1200px) rotateX(4deg) translateY(-8px); }
        }
        @keyframes shimmer {
          0%   { background-position: -200% center; }
          100% { background-position: 200% center; }
        }
        @keyframes borderGlow {
          0%, 100% { opacity: 0.5; }
          50%       { opacity: 1; }
        }

        .orb1 { animation: orbDrift1 18s ease-in-out infinite; }
        .orb2 { animation: orbDrift2 24s ease-in-out infinite; }
        .orb3 { animation: orbDrift3 14s ease-in-out infinite; }

        .fade-up   { animation: fadeUp 0.8s cubic-bezier(0.16,1,0.3,1) both; }
        .fd1 { animation-delay: 0.05s; }
        .fd2 { animation-delay: 0.18s; }
        .fd3 { animation-delay: 0.30s; }
        .fd4 { animation-delay: 0.44s; }
        .fd5 { animation-delay: 0.58s; }
        .fd6 { animation-delay: 0.72s; }

        .hero-img {
          animation: imgFloat 6s ease-in-out infinite;
          transform: perspective(1200px) rotateX(4deg);
        }
        .screenshot-glow {
          box-shadow:
            0 0 0 1px rgba(255,255,255,0.07),
            0 40px 80px rgba(0,0,0,0.7),
            0 0 120px rgba(220,38,38,0.08);
        }
        .shimmer-badge {
          background: linear-gradient(90deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.12) 50%, rgba(255,255,255,0.06) 100%);
          background-size: 200% auto;
          animation: shimmer 3s linear infinite;
        }
        .glow-btn {
          box-shadow: 0 0 0 1px rgba(220,38,38,0.4), 0 8px 32px rgba(220,38,38,0.25);
          transition: box-shadow 0.25s, transform 0.2s;
        }
        .glow-btn:hover {
          box-shadow: 0 0 0 1px rgba(220,38,38,0.6), 0 12px 40px rgba(220,38,38,0.4);
          transform: translateY(-1px);
        }
        .secondary-btn {
          transition: background 0.2s, color 0.2s;
        }
        .secondary-btn:hover { background: rgba(255,255,255,0.08); }
        .feature-card {
          transition: border-color 0.3s;
        }
        .feature-card:hover { border-color: rgba(255,255,255,0.12); }
        .gradient-text {
          background: linear-gradient(135deg, #f97316, #dc2626);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        .section-img {
          transition: transform 0.5s ease, box-shadow 0.5s ease;
        }
        .section-img:hover {
          transform: scale(1.015);
          box-shadow:
            0 0 0 1px rgba(255,255,255,0.1),
            0 48px 96px rgba(0,0,0,0.8),
            0 0 140px rgba(220,38,38,0.12);
        }
      `}</style>

      {/* ── Ambient background orbs ── */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none" style={{ zIndex: 0 }}>
        <div className="orb1 absolute" style={{ top: '-10%', left: '20%', width: 600, height: 600, borderRadius: '50%', background: 'radial-gradient(circle, rgba(194,65,12,0.18) 0%, transparent 70%)', filter: 'blur(1px)' }} />
        <div className="orb2 absolute" style={{ top: '30%', right: '-5%', width: 500, height: 500, borderRadius: '50%', background: 'radial-gradient(circle, rgba(220,38,38,0.12) 0%, transparent 70%)', filter: 'blur(1px)' }} />
        <div className="orb3 absolute" style={{ bottom: '10%', left: '5%', width: 400, height: 400, borderRadius: '50%', background: 'radial-gradient(circle, rgba(120,30,10,0.14) 0%, transparent 70%)', filter: 'blur(1px)' }} />
        {/* Subtle grid */}
        <div className="absolute inset-0" style={{
          backgroundImage: 'linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)',
          backgroundSize: '80px 80px',
        }} />
      </div>

      <div className="relative" style={{ zIndex: 1 }}>

        {/* ── Navbar ── */}
        <nav className="sticky top-0 z-50 flex items-center justify-between px-6 md:px-12 py-4" style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(20px)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
          <button onClick={() => {}} className="flex items-center gap-2.5">
            <img src="/gemini-svg.svg" alt="PyroWatch" className="w-8 h-8 rounded-lg object-cover" />
            <span style={{ fontFamily: '"Outfit", sans-serif', fontWeight: 700, fontSize: 16, letterSpacing: '0.04em' }}>
              PYROWATCH
            </span>
          </button>
          <div className="flex items-center gap-3">
            <button
              onClick={() => onNavigate('overview')}
              className="secondary-btn text-[12px] font-medium text-white/60 hover:text-white px-4 py-2 rounded-lg"
            >
              Dashboard
            </button>
            <button
              onClick={() => onNavigate('fire-map')}
              className="glow-btn flex items-center gap-2 px-4 py-2 rounded-lg text-[12px] font-semibold text-white"
              style={{ background: 'linear-gradient(135deg, #c2410c, #991b1b)' }}
            >
              <Flame className="w-3.5 h-3.5" />
              Open App
            </button>
          </div>
        </nav>

        {/* ── Hero ── */}
        <section className="flex flex-col items-center text-center px-6 pt-20 pb-0">
          {/* Badge */}
          <div className="fade-up fd1 shimmer-badge inline-flex items-center gap-2 px-4 py-1.5 rounded-full border mb-8" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>
            <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
            <span className="text-[10px] font-mono tracking-widest text-white/60 uppercase">Wildfire Intelligence Platform · Active</span>
          </div>

          {/* Headline */}
          <h1 className="fade-up fd2 leading-[1.0] mb-6" style={{ fontFamily: '"Outfit", sans-serif', fontWeight: 900, fontSize: 'clamp(42px, 8vw, 88px)' }}>
            AI-Powered Wildfire<br />
            <span className="gradient-text">Detection & Reporting</span>
          </h1>

          {/* Subtext */}
          <p className="fade-up fd3 text-white/50 max-w-xl mb-10 leading-relaxed" style={{ fontSize: 16 }}>
            Satellite data ingestion, XGBoost ignition risk scoring, Rothermel fire spread simulation,
            and Gemini LLM incident reports — all in one automated MLOps pipeline.
          </p>

          {/* CTAs */}
          <div className="fade-up fd4 flex items-center gap-3 mb-16">
            <button
              onClick={() => onNavigate('fire-map')}
              className="glow-btn flex items-center gap-2.5 px-7 py-3.5 rounded-xl text-[14px] font-semibold text-white"
              style={{ background: 'linear-gradient(135deg, #c2410c, #991b1b)' }}
            >
              <Flame className="w-4 h-4" />
              Enter Dashboard
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigate('reports')}
              className="secondary-btn flex items-center gap-2 px-6 py-3.5 rounded-xl text-[14px] font-medium text-white/70 border"
              style={{ borderColor: 'rgba(255,255,255,0.1)' }}
            >
              View Reports
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          {/* Tech pills */}
          <div className="fade-up fd5 flex flex-wrap justify-center gap-2 mb-16 max-w-2xl">
            {PILL_TAGS.map(tag => (
              <span key={tag} className="text-[10px] font-mono text-white/30 px-3 py-1 rounded-full border" style={{ borderColor: 'rgba(255,255,255,0.07)', background: 'rgba(255,255,255,0.02)' }}>
                {tag}
              </span>
            ))}
          </div>

          {/* Hero screenshot */}
          <div className="fade-up fd6 w-full max-w-5xl mx-auto px-0 md:px-6">
            <div className="relative">
              {/* Glow underneath */}
              <div className="absolute -bottom-16 left-1/2 -translate-x-1/2 w-3/4 h-32 rounded-full" style={{ background: 'radial-gradient(ellipse, rgba(220,38,38,0.25) 0%, transparent 70%)', filter: 'blur(20px)' }} />
              {/* Browser chrome */}
              <div className="rounded-xl overflow-hidden screenshot-glow hero-img" style={{ border: '1px solid rgba(255,255,255,0.08)' }}>
                <div className="flex items-center gap-2 px-4 py-3" style={{ background: 'rgba(255,255,255,0.04)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <span className="w-3 h-3 rounded-full" style={{ background: '#ff5f57' }} />
                  <span className="w-3 h-3 rounded-full" style={{ background: '#febc2e' }} />
                  <span className="w-3 h-3 rounded-full" style={{ background: '#28c840' }} />
                  <span className="mx-auto text-[10px] font-mono text-white/20">pyrowatch · fire-detection-map</span>
                </div>
                <img src="/Firemap.png" alt="PyroWatch Fire Detection Map" className="w-full block" style={{ display: 'block' }} />
              </div>
            </div>
          </div>
        </section>

        {/* ── Gradient fade from hero into features ── */}
        <div className="h-48 -mt-1" style={{ background: 'linear-gradient(to bottom, transparent, #000000)' }} />

        {/* ── Feature 1: Real-time monitoring ── */}
        <section className="px-6 md:px-16 py-24 max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-center">
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full mb-6 text-[10px] font-mono tracking-widest uppercase" style={{ background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.2)', color: '#60a5fa' }}>
                <Map className="w-3 h-3" /> Real-Time Monitoring
              </div>
              <h2 className="mb-5 leading-tight" style={{ fontFamily: '"Outfit", sans-serif', fontWeight: 800, fontSize: 'clamp(28px, 4vw, 44px)' }}>
                Live risk scoring across every cell
              </h2>
              <p className="text-white/50 leading-relaxed mb-8" style={{ fontSize: 14 }}>
                H3 hexagonal grid at 64km resolution covering California and Texas.
                OBJ-1 XGBoost + LightGBM ensemble computes ignition probability
                from NOAA VIIRS satellite feeds and ERA5 reanalysis data — updated every 6 hours.
              </p>
              <ul className="space-y-3">
                {['OBJ-2 Rothermel fire spread simulation with Monte Carlo (N=100)', 'Wind vectors, crown fire indicators, and fuel model overlays', 'Click any cell to inspect weather, terrain, canopy, and spread data'].map(item => (
                  <li key={item} className="flex items-start gap-3 text-[13px] text-white/50">
                    <span className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0 mt-1.5" />
                    {item}
                  </li>
                ))}
              </ul>
              <button
                onClick={() => onNavigate('fire-map')}
                className="secondary-btn mt-8 inline-flex items-center gap-2 text-[13px] font-medium text-white/70 hover:text-white"
              >
                Open Fire Map <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Screenshot */}
            <div className="relative">
              <div className="absolute -inset-4 rounded-2xl" style={{ background: 'radial-gradient(ellipse at center, rgba(59,130,246,0.08) 0%, transparent 70%)' }} />
              <div className="relative section-img rounded-xl overflow-hidden" style={{ border: '1px solid rgba(255,255,255,0.07)', boxShadow: '0 32px 80px rgba(0,0,0,0.6)' }}>
                <img src="/Firemap.png" alt="Fire Detection Map" className="w-full block" />
              </div>
            </div>
          </div>
        </section>

        {/* ── Feature 2: AI Reports ── */}
        <section className="px-6 md:px-16 py-24 max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-center">
            {/* Screenshot first on desktop */}
            <div className="relative md:order-first order-last">
              <div className="absolute -inset-4 rounded-2xl" style={{ background: 'radial-gradient(ellipse at center, rgba(234,88,12,0.08) 0%, transparent 70%)' }} />
              <div className="relative section-img rounded-xl overflow-hidden" style={{ border: '1px solid rgba(255,255,255,0.07)', boxShadow: '0 32px 80px rgba(0,0,0,0.6)' }}>
                <img src="/reports.png" alt="Incident Reports" className="w-full block" />
              </div>
            </div>

            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full mb-6 text-[10px] font-mono tracking-widest uppercase" style={{ background: 'rgba(234,88,12,0.1)', border: '1px solid rgba(234,88,12,0.2)', color: '#fb923c' }}>
                <FileText className="w-3 h-3" /> AI-Generated Reports
              </div>
              <h2 className="mb-5 leading-tight" style={{ fontFamily: '"Outfit", sans-serif', fontWeight: 800, fontSize: 'clamp(28px, 4vw, 44px)' }}>
                ICS-209 reports written by Gemini
              </h2>
              <p className="text-white/50 leading-relaxed mb-8" style={{ fontSize: 14 }}>
                OBJ-3 GeminiDisasterReporter synthesises risk data from across the grid
                into structured, ICS-209 aligned incident briefs — validated by Pydantic
                with confidence scores, grounding sources, and escalation triggers.
              </p>
              <ul className="space-y-3">
                {['Gemini 2.5 Flash on Vertex AI with structured output', 'Contributing factors, preventive recommendations, escalation triggers', 'Data completeness badges and human-review flags'].map(item => (
                  <li key={item} className="flex items-start gap-3 text-[13px] text-white/50">
                    <span className="w-1.5 h-1.5 rounded-full bg-orange-500 flex-shrink-0 mt-1.5" />
                    {item}
                  </li>
                ))}
              </ul>
              <button
                onClick={() => onNavigate('reports')}
                className="secondary-btn mt-8 inline-flex items-center gap-2 text-[13px] font-medium text-white/70 hover:text-white"
              >
                View Reports <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </section>

        {/* ── Stats strip ── */}
        <section className="px-6 py-16 max-w-5xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-px rounded-2xl overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
            {[
              { val: '64 km', label: 'H3 Grid Resolution', sub: 'Hexagonal cells' },
              { val: 'OBJ 1·2·3', label: 'ML Pipeline Stages', sub: 'Ignition · Spread · Report' },
              { val: '30 min', label: 'Update Cadence', sub: 'Automated Airflow DAG' },
              { val: 'CA + TX', label: 'Coverage Area', sub: '2 US states' },
            ].map(s => (
              <div key={s.label} className="flex flex-col gap-1.5 px-8 py-8" style={{ background: '#080808' }}>
                <div className="gradient-text font-bold leading-none" style={{ fontFamily: '"Outfit", sans-serif', fontSize: 28 }}>{s.val}</div>
                <div className="text-white/70 font-medium text-[13px]">{s.label}</div>
                <div className="text-white/25 font-mono text-[10px]">{s.sub}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── Pipeline section ── */}
        <section className="px-6 py-20 max-w-5xl mx-auto text-center">
          <p className="text-white/30 font-mono text-[10px] tracking-widest uppercase mb-4">Built on</p>
          <div className="flex flex-wrap justify-center gap-3">
            {[
              { icon: Zap, label: 'Apache Airflow', color: '#16a34a' },
              { icon: Shield, label: 'Vertex AI', color: '#3b82f6' },
              { icon: Activity, label: 'XGBoost · LightGBM', color: '#ea580c' },
              { icon: FileText, label: 'Gemini 2.5 Flash', color: '#7c3aed' },
              { icon: Map, label: 'H3 + Leaflet', color: '#0891b2' },
            ].map(t => {
              const Icon = t.icon;
              return (
                <div key={t.label} className="flex items-center gap-2.5 px-5 py-3 rounded-xl text-[12px] font-medium text-white/50" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <Icon className="w-3.5 h-3.5" style={{ color: t.color }} />
                  {t.label}
                </div>
              );
            })}
          </div>
        </section>

        {/* ── Bottom CTA ── */}
        <section className="px-6 py-32 flex flex-col items-center text-center">
          <div className="absolute left-1/2 -translate-x-1/2" style={{ width: 600, height: 300, background: 'radial-gradient(ellipse, rgba(194,65,12,0.18) 0%, transparent 70%)', filter: 'blur(40px)', pointerEvents: 'none' }} />
          <h2 className="relative mb-6 leading-tight" style={{ fontFamily: '"Outfit", sans-serif', fontWeight: 900, fontSize: 'clamp(36px, 6vw, 68px)' }}>
            Detect fires before<br />they spread.
          </h2>
          <p className="relative text-white/40 mb-10 max-w-md leading-relaxed" style={{ fontSize: 15 }}>
            Open the dashboard to monitor live risk scores, run spread simulations, and generate AI incident reports.
          </p>
          <button
            onClick={() => onNavigate('fire-map')}
            className="relative glow-btn flex items-center gap-3 px-9 py-4 rounded-xl text-[15px] font-semibold text-white"
            style={{ background: 'linear-gradient(135deg, #c2410c, #991b1b)', fontFamily: '"DM Sans", sans-serif' }}
          >
            <Flame className="w-5 h-5" />
            Enter PyroWatch
            <ArrowRight className="w-5 h-5" />
          </button>
        </section>

        {/* ── Footer ── */}
        <footer className="px-6 md:px-16 py-8 flex items-center justify-between flex-wrap gap-4" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
          <div className="flex items-center gap-2.5">
            <img src="/gemini-svg.svg" alt="PyroWatch" className="w-6 h-6 rounded-md object-cover" />
            <span className="text-white/30 font-mono text-[10px] tracking-widest uppercase">PyroWatch · MLOps Capstone · 2026</span>
          </div>
          <div className="flex items-center gap-6">
            {['Fire Map', 'Reports', 'Risk Monitor', 'Pipeline'].map((l, i) => {
              const views = ['fire-map', 'reports', 'risk-monitor', 'data-pipeline'];
              return (
                <button key={l} onClick={() => onNavigate(views[i])} className="text-white/25 hover:text-white/60 transition-colors font-mono text-[10px] tracking-wide">
                  {l}
                </button>
              );
            })}
          </div>
        </footer>

      </div>
    </div>
  );
}
