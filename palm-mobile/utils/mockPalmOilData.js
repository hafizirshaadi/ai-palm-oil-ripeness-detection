/**
 * Mock Palm Oil Data
 * This file simulates the data that would be returned from the Commodities API
 * All prices are in Malaysian Ringgit (MYR)
 */

// Current timestamp (simulated)
const currentTimestamp = Math.floor(Date.now() / 1000);

// Generate a date in YYYY-MM-DD format based on days ago
const getDate = (daysAgo = 0) => {
  const date = new Date();
  date.setDate(date.getDate() - daysAgo);
  return date.toISOString().split('T')[0];
};

// Latest palm oil price data
export const latestData = {
  success: true,
  timestamp: currentTimestamp,
  base: "MYR",
  date: getDate(0),
  rates: {
    "PALMOIL": 0.000245, // Rate is inverted as per API convention (1/rate = price)
    "MYR": 1
  },
  unit: "per metric ton"
};

// Historical price data for the past 5 months
export const timeSeriesData = {
  success: true,
  timeseries: true,
  start_date: getDate(150),
  end_date: getDate(1),
  base: "MYR",
  rates: {
    [getDate(150)]: { "PALMOIL": 0.000276, "MYR": 1 }, // ~3,623 MYR
    [getDate(120)]: { "PALMOIL": 0.000268, "MYR": 1 }, // ~3,731 MYR
    [getDate(90)]: { "PALMOIL": 0.000259, "MYR": 1 },  // ~3,861 MYR
    [getDate(60)]: { "PALMOIL": 0.000251, "MYR": 1 },  // ~3,984 MYR
    [getDate(30)]: { "PALMOIL": 0.000247, "MYR": 1 },  // ~4,049 MYR
    [getDate(15)]: { "PALMOIL": 0.000246, "MYR": 1 },  // ~4,065 MYR
    [getDate(7)]: { "PALMOIL": 0.000245, "MYR": 1 },   // ~4,082 MYR
    [getDate(1)]: { "PALMOIL": 0.000245, "MYR": 1 },   // ~4,082 MYR
  }
};

// Monthly data for calculating averages
export const generateMonthlyData = () => {
  const months = ['January', 'February', 'March', 'April', 'May'];
  const monthlyData = {};
  
  // Generate dates for the last 5 months
  months.forEach((month, index) => {
    const date = new Date();
    date.setMonth(date.getMonth() - (4 - index)); // 4, 3, 2, 1, 0 months ago
    
    const monthYear = `${month} ${date.getFullYear()}`;
    monthlyData[monthYear] = [];
    
    // Generate 20-25 data points per month with slight variations
    const baseRate = 0.000276 - (index * 0.000008); // Decreasing rate (increasing price)
    const numDays = 20 + Math.floor(Math.random() * 6); // 20-25 days
    
    for (let i = 0; i < numDays; i++) {
      // Add small fluctuations to the rate
      const variation = (Math.random() - 0.5) * 0.000002;
      monthlyData[monthYear].push(baseRate + variation);
    }
  });
  
  return monthlyData;
};

// Processed data in the format expected by the application
export const processedData = {
  current: 4082.45, // 1/0.000245 ≈ 4,082 MYR
  currency: 'MYR',
  unit: 'per metric ton',
  lastUpdated: new Date().toLocaleDateString('en-MY', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  }),
  historicalData: {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    datasets: [
      {
        data: [3623.19, 3731.34, 3861.00, 4049.72, 4082.45],
        color: (opacity = 1) => `rgba(46, 139, 87, ${opacity})`,
        strokeWidth: 2
      }
    ]
  },
  monthlyAverages: [
    { month: 'January', price: 3623.19 },
    { month: 'February', price: 3731.34 },
    { month: 'March', price: 3861.00 },
    { month: 'April', price: 4049.72 },
    { month: 'May', price: 4082.45 }
  ],
  priceChange: {
    daily: 0.4, // 0.4% increase from yesterday
    weekly: 0.8, // 0.8% increase from last week
    monthly: 12.5, // 12.5% increase from last month
  }
};
