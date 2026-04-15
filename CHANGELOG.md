# Frontend Changelog

## [2026-04-14] Responsive Mobile Support & Viewer Mode

### Added
- **`src/App.jsx`** — Viewer mode (`?mode=viewer`)
  - URL parameter `?mode=viewer` restricts access to Fire Detection Map and Incident Reports only
  - No sidebar, no pipeline controls, no edit capabilities
  - Simple top bar with tab switcher for viewers
- **`qrcode.png`** — QR code for quick mobile/tablet access via local network

### Changed
- **`src/components/layout/Header.jsx`** — Dynamic dates
  - System Overview subtitle: replaced hardcoded `Jan 2025` with dynamic system date
  - Wildfire Risk Monitor subtitle: same fix, now displays current month & year
  - Header clock (top-right): replaced hardcoded `2025-01-15T18:04:32Z` with live system time

- **`src/components/fire-map/FireMap.jsx`** — Mobile responsive & click fix
  - Cell Detail panel: added visible Close button (X icon + "Close" text)
  - Cell Detail panel: on mobile, slides in as overlay from right instead of fixed side panel
  - Added backdrop overlay on mobile — tap outside panel to close
  - LayerBar: responsive width (`right-3` on mobile, `right-[290px]` on desktop) with `flex-wrap`

### Fixed
- **`src/components/fire-map/FireMap.jsx`** — Critical cells hard to click
  - Fire pulse rings, spread burn overlays, ignition pulses, and spread arrows were intercepting click events on top of critical/fire-detected cells
  - Added `interactive={false}` to all decorative overlay layers so clicks pass through to the hex cell polygons underneath

- **`src/components/fire-map/FireMap.jsx`** — LayerBar overflow on mobile
  - Wind/Crown/Spread buttons were overflowing the toolbar on small screens
  - Inner button groups now use `flex-wrap` and `flex-shrink-0` to wrap properly
  - Added `whitespace-nowrap` to prevent button text from being cut off
  - Reduced padding on mobile (`px-1.5`) while keeping desktop size (`md:px-2.5`)
  - Separator borders hidden on mobile (`md:border-l`) to save space

- **`src/components/reports/IncidentReports.jsx`** — Mobile responsive
  - Header strip: stacks vertically on mobile (`flex-col` → `flex-row` at `md`)
  - Header metadata: 2-column grid on mobile, inline flex on desktop
  - Report card header: stacks content and risk badges vertically on mobile
  - Expanded content grids: single column on mobile, multi-column on desktop
  - Footer and validation sections: flex-wrap for small screens

### Why
- Dates were hardcoded to January 2025, now reflect current system time
- Fire Map and Incident Reports were unusable on mobile — cells invisible, panels overlapping, content overflowing
- Viewer mode enables safe sharing via QR code — viewers can only browse maps and reports without modifying data
