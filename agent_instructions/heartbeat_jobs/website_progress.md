# Website Builder Instructions — EnGem Revamp Prompt (v9.1)

## Prompt Changelog
- 2026-03-02 — agent: v9.1 UI enhancements; added CSS animations, refined hover states, and expanded section content for Features, Use Cases, Docs, and Blog.
- 2026-03-02 — agent: Deployed V9.0 Omni-EnGem; full reconstruction with 6 mandatory sections (Home, Features, Use Cases, Docs, Blog, Contact).
- 2026-03-02 — agent: Fixed submodule deployment and synchronized skills snippet.
- 2026-03-01 — agent: Rebranding EnGem to a "Gemini Blue" aesthetic.
- 2026-03-01 — agent: Major UI/UX overhaul; implemented glassmorphism and premium typography.

## Core Website Requirements
1. **Home Page**: Welcoming intro to EnGem.
2. **Features**: Key highlights (text generation, context understanding, tasks).
3. **Use Cases**: Industry-specific examples.
4. **Documentation**: Installation, API refs, tutorials.
5. **Contact**: Form and info.
6. **Blog**: Articles and updates.

## Design & Tone
- **Aesthetic**: "Gemini Blue" (Deep blacks, vibrant blues, glassmorphism).
- **Layout**: Bento-grid style for feature highlights.
- **Typography**: Premium, high-readability fonts (Inter/Outfit).
- **Interactivity**: Smooth animations (CSS @keyframes) and scroll-reveal effects.

## Self-Editing Rules
- **Formatting**: <pre> tags must ALWAYS be siblings to <p> tags. Never nest <pre> inside <p>.
- **Logic**: Dynamic sections (Skills) must link to skills/ scripts.
- **Responsive**: Mobile-first design excellence.

## Deployment Protocol
1. **Submodule**: Update, commit, and push dylpickle13.github.io to main, master, and gh-pages.
2. **Root Repo**: Update pointer, add instruction files (force-add heartbeat), commit, and push to all branches.
