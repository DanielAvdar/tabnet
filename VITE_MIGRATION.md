# Vite Migration Documentation

## Overview

This document outlines the migration of the TabNet project to include a modern React web interface built with Vite as the build tool.

## Migration Steps Completed

### 1. Project Structure Setup
- Created `web/` directory for the React application
- Set up standard React project structure with `src/`, `public/`, and configuration files

### 2. Vite Configuration
- **File Created**: `web/vite.config.js`
- **Features Configured**:
  - React plugin with Fast Refresh support
  - Development server on port 3000 with auto-open
  - Production builds with source maps
  - Optimized build output directory

### 3. Package Dependencies
- **File Created**: `web/package.json`
- **Dependencies Added**:
  - `react` and `react-dom` (^18.2.0) - Core React libraries
  - `vite` (^5.2.0) - Build tool and development server
  - `@vitejs/plugin-react` (^4.2.1) - Vite React plugin
  - ESLint and React-specific linting plugins

### 4. Scripts Configuration
Updated `package.json` scripts to use Vite:
- `npm run dev` - Starts Vite development server
- `npm run build` - Creates optimized production build
- `npm run preview` - Previews production build locally
- `npm run lint` - Runs ESLint for code quality

### 5. Entry Point and HTML Template
- **HTML Template**: `web/index.html` - Modern HTML5 template with React root
- **React Entry Point**: `web/src/main.jsx` - React application bootstrap
- **Main Component**: `web/src/App.jsx` - Sample React component with TabNet branding

### 6. Development Environment
- **ESLint Configuration**: `web/.eslintrc.cjs` - Code quality and React best practices
- **Gitignore**: `web/.gitignore` - Excludes node_modules, dist, and temporary files
- **Styling**: CSS files with modern styling and dark/light theme support

## Key Benefits of Vite Migration

### Development Experience
- **Instant Server Start**: Native ES modules eliminate bundling during development
- **Fast HMR**: React Fast Refresh provides superior hot module replacement
- **Modern JavaScript**: Native ES6+ support without transpilation overhead
- **TypeScript Ready**: Easy migration to TypeScript by renaming files to `.tsx`

### Build Performance
- **Rollup-based Builds**: Optimized production builds with tree-shaking
- **Code Splitting**: Automatic code splitting for better loading performance
- **Asset Optimization**: Built-in asset optimization and compression
- **Source Maps**: Development-friendly debugging with source maps

### Developer Productivity
- **Zero Configuration**: Works out of the box with sensible defaults
- **Plugin Ecosystem**: Rich plugin ecosystem for extending functionality
- **Modern CSS**: Native CSS modules and PostCSS support
- **Framework Agnostic**: Easy to integrate with any UI library or framework

## File Structure

```
web/
├── public/                 # Static assets
│   ├── vite.svg           # Vite logo
│   └── index.html         # Main HTML template (NOTE: Moved to root by Vite)
├── src/                   # Source code
│   ├── assets/           # React assets (images, icons)
│   │   └── react.svg     # React logo
│   ├── App.jsx           # Main App component
│   ├── App.css           # App-specific styles
│   ├── main.jsx          # React entry point
│   └── index.css         # Global styles
├── .eslintrc.cjs         # ESLint configuration
├── .gitignore            # Git ignore rules
├── package.json          # Dependencies and scripts
├── README.md             # Web project documentation
└── vite.config.js        # Vite configuration
```

## Usage Instructions

### Prerequisites
- Node.js version 18 or higher
- npm or yarn package manager

### Getting Started
1. Navigate to the web directory:
   ```bash
   cd web
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start development server:
   ```bash
   npm run dev
   ```
   This will start the server at `http://localhost:3000`

4. Build for production:
   ```bash
   npm run build
   ```
   Built files will be in the `dist/` directory

5. Preview production build:
   ```bash
   npm run preview
   ```

### Code Quality
Run linting to maintain code quality:
```bash
npm run lint
```

## Integration with TabNet

The web interface is designed to complement the existing TabNet Python library. Future development can include:

- **Model Visualization**: Interactive displays of TabNet attention masks and feature importance
- **Data Upload Interface**: Web-based data input for TabNet model inference
- **Training Dashboard**: Real-time monitoring of TabNet model training progress
- **Results Visualization**: Charts and graphs for model performance metrics

## Maintenance Notes

### Dependency Updates
- Keep Vite and React dependencies updated for security and performance
- Use `npm audit` to check for vulnerabilities
- Test thoroughly after major version updates

### Configuration Customization
- The `vite.config.js` file can be extended for additional features
- ESLint rules can be customized in `.eslintrc.cjs`
- Add TypeScript support by installing types and renaming files to `.tsx`

### Build Optimization
- The build is already optimized with code splitting and tree-shaking
- Consider adding a CDN for static asset delivery in production
- Monitor bundle size with Vite's built-in analysis tools

## No Breaking Changes

This migration adds new functionality without affecting the existing Python TabNet library:
- All existing Python code remains unchanged
- Documentation and examples continue to work as before
- The web interface is completely optional and self-contained
- Build and test processes for the Python library are unaffected
