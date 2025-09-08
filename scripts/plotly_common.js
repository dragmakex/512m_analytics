/**
 * Common utilities and configurations for all Plotly.js charts
 * 
 * This module provides shared styling, themes, and utility functions
 * to ensure consistency across all chart implementations.
 */

const PlotlyCommon = (function() {
    'use strict';
    
    // Theme colors matching Python matplotlib style
    const THEME_PALETTE = ['#f7f3ec', '#ede4da', '#b9a58f', '#574c40', '#36312a'];
    
    const MUTED_BLUES = [
        '#2b3e50', '#3c5a77', '#4f7192', '#5f86a8', '#6f9bbd',
        '#86abc7', '#9bbad1', '#afc8da', '#c3d5e3', '#d7e2ec'
    ];
    
    // Display names for pools
    const DISPLAY_POOL_NAMES = {
        '0': 'Ethena sUSDe',
        '1': 'Maple USDC',
        '2': 'Sky sUSDS',
        '3': 'AAVE USDT',
        '4': 'Morpho Spark USDC',
        '5': 'Sky DSR DAI',
        '6': 'Usual USD0++',
        '10': 'Morpho USUALUSDC+',
        '13': 'Fluid USDC'
    };
    
    // Common layout configuration
    const getBaseLayout = (title = '') => ({
        title: {
            text: title,
            x: 0.5,
            font: { family: 'serif', size: 14, color: '#333' }
        },
        plot_bgcolor: THEME_PALETTE[0],
        paper_bgcolor: THEME_PALETTE[0],
        font: {
            family: 'serif',
            size: 10,
            color: '#333'
        },
        margin: { l: 80, r: 50, t: 80, b: 80 },
        showlegend: true,
        legend: {
            font: { size: 9, family: 'serif' },
            bgcolor: 'rgba(247,243,236,0.8)',
            bordercolor: 'rgba(0,0,0,0.2)',
            borderwidth: 1
        }
    });
    
    // Axis styling
    const getAxisStyle = (title = '') => ({
        title: { text: title, font: { family: 'serif', size: 11 } },
        showgrid: true,
        gridwidth: 0.5,
        gridcolor: 'rgba(0,0,0,0.3)',
        tickfont: { size: 9, family: 'serif' },
        showline: true,
        linewidth: 0.8,
        linecolor: '#333'
    });
    
    // Common configuration for all charts
    const getConfig = () => ({
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d'],
        displaylogo: false,
        toImageButtonOptions: {
            format: 'png',
            filename: 'defi_chart',
            height: 600,
            width: 1200,
            scale: 1
        }
    });
    
    // Logo watermark configuration
    const getLogoImage = (logoUrl) => ({
        source: logoUrl,
        xref: "paper",
        yref: "paper", 
        x: 0.5,
        y: 0.5,
        sizex: 0.25,
        sizey: 0.25,
        xanchor: "center",
        yanchor: "middle",
        opacity: 0.05,
        layer: "below"
    });
    
    // Utility function to format numbers
    const formatNumber = (num, decimals = 4) => {
        if (num === null || num === undefined) return 'N/A';
        return num.toFixed(decimals);
    };
    
    // Utility function to format large numbers (TVL)
    const formatTVL = (value) => {
        if (value >= 1e9) return `$${(value/1e9).toFixed(2)}B`;
        if (value >= 1e6) return `$${(value/1e6).toFixed(2)}M`;
        if (value >= 1e3) return `$${(value/1e3).toFixed(2)}K`;
        return `$${value.toFixed(2)}`;
    };
    
    // Date formatting for x-axis
    const getDateFormat = () => ({
        tickformat: '%Y-%m-%d',
        tickangle: -45
    });
    
    // Error handling wrapper
    const safeExecute = async (func, errorMessage = 'Chart error') => {
        try {
            return await func();
        } catch (error) {
            console.error(`${errorMessage}:`, error);
            return null;
        }
    };
    
    // Data fetching with retry logic
    const fetchWithRetry = async (url, maxRetries = 3, delay = 1000) => {
        for (let i = 0; i < maxRetries; i++) {
            try {
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return await response.json();
            } catch (error) {
                console.error(`Fetch attempt ${i + 1} failed:`, error);
                if (i === maxRetries - 1) throw error;
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    };
    
    // Show loading indicator
    const showLoading = (containerId) => {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; font-family: serif;">
                    <div style="font-size: 18px; color: ${THEME_PALETTE[3]};">Loading chart data...</div>
                    <div style="margin-top: 20px;">
                        <div style="display: inline-block; width: 40px; height: 40px; border: 3px solid ${THEME_PALETTE[1]}; 
                                    border-radius: 50%; border-top-color: ${THEME_PALETTE[3]}; animation: spin 1s linear infinite;"></div>
                    </div>
                </div>
                <style>
                    @keyframes spin {
                        to { transform: rotate(360deg); }
                    }
                </style>
            `;
        }
    };
    
    // Show error message
    const showError = (containerId, message) => {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; color: #dc3545; background: #f8f9fa; 
                            border: 1px solid #dee2e6; border-radius: 8px; font-family: serif;">
                    <div style="font-size: 16px; margin-bottom: 10px;">⚠️ Chart Error</div>
                    <div style="font-size: 12px;">${message}</div>
                    <div style="font-size: 10px; margin-top: 10px; color: #6c757d;">
                        Please check the console for more details.
                    </div>
                </div>
            `;
        }
    };
    
    // Convert Python DataFrame-like data to arrays for Plotly
    const extractDataArrays = (poolData) => {
        const dates = Object.keys(poolData).sort();
        const result = {
            dates: dates.map(d => new Date(d)),
            data: {}
        };
        
        // Extract all column names from the first data point
        if (dates.length > 0) {
            const columns = Object.keys(poolData[dates[0]]);
            columns.forEach(col => {
                result.data[col] = dates.map(date => poolData[date][col]);
            });
        }
        
        return result;
    };
    
    // Calculate moving average
    const movingAverage = (data, window) => {
        const result = [];
        for (let i = 0; i < data.length; i++) {
            if (i < window - 1) {
                result.push(null);
            } else {
                const sum = data.slice(i - window + 1, i + 1).reduce((a, b) => a + b, 0);
                result.push(sum / window);
            }
        }
        return result;
    };
    
    // Calculate rolling correlation
    const rollingCorrelation = (x, y, window) => {
        const result = [];
        for (let i = 0; i < x.length; i++) {
            if (i < window - 1) {
                result.push(null);
            } else {
                const xWindow = x.slice(i - window + 1, i + 1);
                const yWindow = y.slice(i - window + 1, i + 1);
                const correlation = calculateCorrelation(xWindow, yWindow);
                result.push(correlation);
            }
        }
        return result;
    };
    
    // Calculate correlation coefficient
    const calculateCorrelation = (x, y) => {
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((total, xi, i) => total + xi * y[i], 0);
        const sumX2 = x.reduce((total, xi) => total + xi * xi, 0);
        const sumY2 = y.reduce((total, yi) => total + yi * yi, 0);
        
        const num = n * sumXY - sumX * sumY;
        const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        return den === 0 ? 0 : num / den;
    };
    
    // Export public API
    return {
        THEME_PALETTE,
        MUTED_BLUES,
        DISPLAY_POOL_NAMES,
        getBaseLayout,
        getAxisStyle,
        getConfig,
        getLogoImage,
        formatNumber,
        formatTVL,
        getDateFormat,
        safeExecute,
        fetchWithRetry,
        showLoading,
        showError,
        extractDataArrays,
        movingAverage,
        rollingCorrelation,
        calculateCorrelation
    };
})();

// Make it available globally
window.PlotlyCommon = PlotlyCommon;