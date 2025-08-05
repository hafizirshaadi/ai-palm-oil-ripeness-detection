// Palm Oil Price Service - Simulated data based on Commodities API structure
// Uses mock data to simulate real API calls

import { 
  latestData, 
  timeSeriesData, 
  processedData, 
  generateMonthlyData 
} from './mockPalmOilData';

// Palm oil symbol in the simulated API
const PALM_OIL_SYMBOL = 'PALMOIL';

/**
 * Get the latest palm oil price
 * @returns {Promise<Object>} price data
 */
export const getCurrentPalmOilPrice = async () => {
  // Simulate API call delay
  return new Promise((resolve) => {
    setTimeout(() => {
      try {
        // Get the palm oil rate (inverted as per API convention)
        const palmOilRate = latestData.rates[PALM_OIL_SYMBOL];
        const currentPrice = palmOilRate ? (1 / palmOilRate) : 0;
        
        // Format the date from timestamp
        const lastUpdated = new Date(latestData.timestamp * 1000).toLocaleDateString('en-MY', {
          year: 'numeric',
          month: 'short',
          day: 'numeric'
        });
        
        // Get historical data
        const historicalData = processHistoricalData(timeSeriesData);
        
        // Calculate price changes
        const priceChanges = calculatePriceChanges(timeSeriesData);
        
        // Get monthly averages
        const monthlyAverages = calculateMonthlyAverages();
        
        resolve({
          current: parseFloat(currentPrice.toFixed(2)),
          currency: 'MYR',
          unit: 'per metric ton',
          lastUpdated,
          historicalData,
          monthlyAverages,
          priceChange: priceChanges
        });
      } catch (error) {
        console.error('Error processing palm oil price data:', error);
        resolve(processedData); // Fallback to pre-processed data
      }
    }, 800); // Simulate network delay
  });
};

/**
 * Process the time series data into a format suitable for charts
 * @param {Object} data - Raw time series data
 * @returns {Object} Formatted data for historical chart
 */
const processHistoricalData = (data) => {
  try {
    // Get dates in chronological order
    const dates = Object.keys(data.rates).sort((a, b) => new Date(a) - new Date(b));
    
    // Select evenly spaced dates (5 points total)
    const selectedDates = [];
    const step = Math.max(1, Math.floor(dates.length / 5));
    
    for (let i = 0; i < dates.length; i += step) {
      if (selectedDates.length < 5) {
        selectedDates.push(dates[i]);
      }
    }
    
    // If we don't have 5 dates yet, add the most recent date
    if (selectedDates.length < 5 && !selectedDates.includes(dates[dates.length - 1])) {
      selectedDates.push(dates[dates.length - 1]);
    }
    
    const labels = [];
    const prices = [];
    
    // Extract the data for each selected date
    selectedDates.forEach(date => {
      const rate = data.rates[date][PALM_OIL_SYMBOL];
      const price = rate ? (1 / rate) : 0;
      
      // Format the date as a month abbreviation
      const monthAbbr = new Date(date).toLocaleString('default', { month: 'short' });
      
      labels.push(monthAbbr);
      prices.push(parseFloat(price.toFixed(2)));
    });
    
    return {
      labels,
      datasets: [
        {
          data: prices,
          color: (opacity = 1) => `rgba(46, 139, 87, ${opacity})`,
          strokeWidth: 2
        }
      ]
    };
  } catch (error) {
    console.error('Error processing historical data:', error);
    return processedData.historicalData;
  }
};

/**
 * Calculate price changes for different time periods
 * @param {Object} data - Raw time series data
 * @returns {Object} Price change percentages
 */
const calculatePriceChanges = (data) => {
  try {
    // Get the current price
    const currentRate = latestData.rates[PALM_OIL_SYMBOL];
    const currentPrice = 1 / currentRate;
    
    // Get dates in reverse chronological order (newest first)
    const dates = Object.keys(data.rates).sort((a, b) => new Date(b) - new Date(a));
    
    // Calculate daily change (1 day ago)
    let dailyChange = 0;
    if (dates.length > 0) {
      const yesterdayRate = data.rates[dates[0]][PALM_OIL_SYMBOL];
      const yesterdayPrice = 1 / yesterdayRate;
      dailyChange = calculatePercentageChange(yesterdayPrice, currentPrice);
    }
    
    // Calculate weekly change (7 days ago)
    let weeklyChange = 0;
    const weekAgoDate = dates.find(date => {
      const daysDiff = getDaysDifference(new Date(date), new Date());
      return daysDiff >= 7;
    });
    
    if (weekAgoDate) {
      const weekAgoRate = data.rates[weekAgoDate][PALM_OIL_SYMBOL];
      const weekAgoPrice = 1 / weekAgoRate;
      weeklyChange = calculatePercentageChange(weekAgoPrice, currentPrice);
    }
    
    // Calculate monthly change (30 days ago)
    let monthlyChange = 0;
    const monthAgoDate = dates.find(date => {
      const daysDiff = getDaysDifference(new Date(date), new Date());
      return daysDiff >= 30;
    });
    
    if (monthAgoDate) {
      const monthAgoRate = data.rates[monthAgoDate][PALM_OIL_SYMBOL];
      const monthAgoPrice = 1 / monthAgoRate;
      monthlyChange = calculatePercentageChange(monthAgoPrice, currentPrice);
    }
    
    return {
      daily: parseFloat(dailyChange.toFixed(1)),
      weekly: parseFloat(weeklyChange.toFixed(1)),
      monthly: parseFloat(monthlyChange.toFixed(1))
    };
  } catch (error) {
    console.error('Error calculating price changes:', error);
    return processedData.priceChange;
  }
};

/**
 * Calculate monthly average prices
 * @returns {Array} Monthly average price data
 */
const calculateMonthlyAverages = () => {
  try {
    const monthlyRates = generateMonthlyData();
    
    const monthlyAverages = Object.keys(monthlyRates).map(monthYear => {
      const rates = monthlyRates[monthYear];
      // Convert rates to prices and calculate average
      const prices = rates.map(rate => 1 / rate);
      const avgPrice = prices.reduce((sum, price) => sum + price, 0) / prices.length;
      
      return {
        month: monthYear.split(' ')[0], // Just the month name
        price: parseFloat(avgPrice.toFixed(2))
      };
    });
    
    // Sort by chronological order
    return monthlyAverages.sort((a, b) => {
      const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 
                    'August', 'September', 'October', 'November', 'December'];
      return months.indexOf(a.month) - months.indexOf(b.month);
    });
  } catch (error) {
    console.error('Error calculating monthly averages:', error);
    return processedData.monthlyAverages;
  }
};

/**
 * Calculate percentage change between two values
 * @param {number} oldValue - Old value
 * @param {number} newValue - New value
 * @returns {number} percentage change
 */
const calculatePercentageChange = (oldValue, newValue) => {
  if (oldValue === 0) return 0;
  return ((newValue - oldValue) / oldValue) * 100;
};

/**
 * Calculate the difference in days between two dates
 * @param {Date} date1 - First date
 * @param {Date} date2 - Second date
 * @returns {number} Difference in days
 */
const getDaysDifference = (date1, date2) => {
  const diffTime = Math.abs(date2 - date1);
  return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
};

/**
 * Get historical palm oil price data
 * @param {string} startDate - Start date in YYYY-MM-DD format
 * @param {string} endDate - End date in YYYY-MM-DD format
 * @returns {Promise<Object>} historical price data
 */
export const getHistoricalPalmOilPrices = async (startDate, endDate) => {
  // Simulate API call with a delay
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(processHistoricalData(timeSeriesData));
    }, 800);
  });
};

export default {
  getCurrentPalmOilPrice,
  getHistoricalPalmOilPrices,
};
