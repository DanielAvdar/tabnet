# TabNet Web Interface

This is a React web application built with Vite for the TabNet project. It provides a modern development environment with fast HMR (Hot Module Replacement) and optimized builds.

## Getting Started

### Prerequisites

Make sure you have Node.js (version 18 or higher) installed on your system.

### Installation

1. Navigate to the web directory:
   ```bash
   cd web
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

### Development

Start the development server:
```bash
npm run dev
```

This will start the Vite development server, typically at `http://localhost:3000`. The server includes:
- Fast HMR (Hot Module Replacement)
- Automatic browser refresh on changes
- Source maps for debugging

### Building for Production

Create an optimized build:
```bash
npm run build
```

The built files will be in the `dist` directory.

### Preview Production Build

Preview the production build locally:
```bash
npm run preview
```

### Linting

Run ESLint to check code quality:
```bash
npm run lint
```

## Project Structure

```
web/
├── public/           # Static assets
│   ├── vite.svg     # Vite logo
│   └── index.html   # HTML template (moved here by Vite)
├── src/             # Source code
│   ├── assets/      # React assets (images, etc.)
│   ├── App.jsx      # Main App component
│   ├── App.css      # App styles
│   ├── main.jsx     # React entry point
│   └── index.css    # Global styles
├── package.json     # Dependencies and scripts
├── vite.config.js   # Vite configuration
└── .eslintrc.cjs    # ESLint configuration
```

## Migration from Legacy Build Tools

This project has been configured with Vite from the start, providing:

- **Fast Development**: Vite's native ES modules approach for instant server start
- **Optimized Builds**: Rollup-based production builds with code splitting
- **Modern JavaScript**: Native ES modules support with no bundling in development
- **React Fast Refresh**: Better HMR experience than traditional hot reloading
- **TypeScript Support**: Ready for TypeScript if needed (just rename files to `.tsx`)

## Configuration

### Vite Configuration

The `vite.config.js` file includes:
- React plugin for JSX support and Fast Refresh
- Development server configuration (port 3000, auto-open browser)
- Build configuration with source maps enabled

### ESLint Configuration

The project uses ESLint with React-specific rules for code quality and consistency.

## About TabNet

TabNet is an interpretable neural network architecture for tabular data. This web interface provides a modern development environment for building web applications that interact with TabNet models.

For more information about the TabNet library, see the main project README.
