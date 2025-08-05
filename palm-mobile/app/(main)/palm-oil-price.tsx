import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, ActivityIndicator, Dimensions } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { getCurrentPalmOilPrice } from '../../utils/palmOilPriceService';

// Define the price data interface
interface PriceData {
  current: number;
  currency: string;
  unit: string;
  lastUpdated: string;
  historicalData: {
    labels: string[];
    datasets: {
      data: number[];
      color: (opacity: number) => string;
      strokeWidth: number;
    }[];
  };
  monthlyAverages: {
    month: string;
    price: number;
  }[];
  priceChange: {
    daily: number;
    weekly: number;
    monthly: number;
  };
}

// Enhanced SimpleBarChart component with gradient bars
interface SimpleBarChartProps {
  data: number[];
  labels: string[];
  width: number;
  height: number;
}

const SimpleBarChart: React.FC<SimpleBarChartProps> = ({ data, labels, width, height }) => {
  const maxValue = Math.max(...data) * 1.1; // Adding 10% padding to the top
  const minValue = Math.min(...data) * 0.9; // Adding buffer at the bottom
  
  return (
    <View style={{ width, height, backgroundColor: 'white', borderRadius: 12, padding: 16, overflow: 'hidden' }}>
      {/* Chart grid lines */}
      <View style={{ position: 'absolute', top: 30, left: 0, right: 0, bottom: 40, zIndex: 1 }}>
        {[0.25, 0.5, 0.75].map((line, i) => (
          <View 
            key={i} 
            style={{ 
              position: 'absolute', 
              top: (height - 70) * (1 - line), 
              left: 16, 
              right: 16, 
              height: 1, 
              backgroundColor: '#eee' 
            }}
          />
        ))}
      </View>
      
      <View style={{ flexDirection: 'row', height: height - 60, alignItems: 'flex-end', paddingBottom: 16, marginTop: 20, zIndex: 2 }}>
        {data.map((value, index) => {
          const barHeight = (value / maxValue) * (height - 80);
          const isHighest = value === Math.max(...data);
          
          return (
            <View key={index} style={{ flex: 1, alignItems: 'center' }}>
              <View style={{
                height: barHeight,
                width: 22,
                borderRadius: 6,
                overflow: 'hidden',
                backgroundColor: isHighest ? '#2e8b57' : '#87c4a3',
                shadowColor: '#000',
                shadowOffset: { width: 0, height: 2 },
                shadowOpacity: 0.1,
                shadowRadius: 2,
                elevation: 2,
              }}>
                <LinearGradient
                  colors={isHighest ? ['#2e8b57', '#1a5733'] : ['#87c4a3', '#5ca77d']}
                  style={{ flex: 1, width: '100%' }}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 0, y: 1 }}
                />
              </View>
              <Text
                style={{
                  fontSize: 11,
                  fontWeight: '500',
                  color: '#555',
                  textAlign: 'center',
                  marginTop: 8,
                }}
              >
                {labels[index].toString()}
              </Text>
              {isHighest && (
                <View style={{ 
                  position: 'absolute', 
                  top: height - 80 - barHeight - 22, 
                  backgroundColor: '#2e8b57', 
                  paddingHorizontal: 6, 
                  paddingVertical: 2, 
                  borderRadius: 10 
                }}>
                  <Text style={{ color: 'white', fontSize: 10, fontWeight: 'bold' }}>
                    {value.toFixed(0)}
                  </Text>
                </View>
              )}
            </View>
          );
        })}
      </View>
      
      {/* Bottom axis line */}
      <View style={{ height: 2, backgroundColor: '#e0e0e0', marginBottom: 5 }} />
      
      {/* Y-axis labels */}
      <View style={{ position: 'absolute', left: 5, top: 5, bottom: 40 }}>
        <Text style={{ position: 'absolute', top: 0, fontSize: 9, color: '#888' }}>
          {Math.round(maxValue).toString()}
        </Text>
        <Text style={{ position: 'absolute', bottom: 0, fontSize: 9, color: '#888' }}>
          {Math.round(minValue).toString()}
        </Text>
      </View>
    </View>
  );
};

export default function PalmOilPricePage() {
  const router = useRouter();
  const [loading, setLoading] = useState<boolean>(true);
  const [priceData, setPriceData] = useState<PriceData | null>(null);
  const screenWidth = Dimensions.get('window').width;

  // Fetch price data using the service
  useEffect(() => {
    const fetchPriceData = async () => {
      try {
        const data = await getCurrentPalmOilPrice();
        setPriceData(data as PriceData);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching palm oil price data:', error);
        setLoading(false);
      }
    };

    fetchPriceData();
  }, []);

  const renderPriceChangeIndicator = (value: number) => {
    const isPositive = value >= 0;
    return (
      <View style={{ flexDirection: 'row', alignItems: 'center' }}>
        {isPositive ? (
          <Ionicons name="arrow-up" size={16} color="green" />
        ) : (
          <Ionicons name="arrow-down" size={16} color="red" />
        )}
        <Text style={{ color: isPositive ? 'green' : 'red', marginLeft: 4 }}>
          <Text>{isPositive ? '+' : ''}</Text>
          <Text>{value.toString()}%</Text>
        </Text>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      {/* Updated Header with Gradient - Same size as Scan page */}
      <LinearGradient
        colors={['#2e8b57', '#1a5733']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 0 }}
        style={styles.headerGradient}
      >
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Palm Oil Price</Text>
          <View style={styles.placeholder}>
            <TouchableOpacity style={styles.infoButton}>
              <Ionicons name="information-circle-outline" size={24} color="white" />
            </TouchableOpacity>
          </View>
        </View>
      </LinearGradient>
      
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2e8b57" />
          <Text style={styles.loadingText}>Loading price data...</Text>
        </View>
      ) : priceData ? (
        <ScrollView 
          style={styles.contentContainer} 
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Enhanced Current Price Card */}
          <View style={styles.priceCard}>
            <View style={styles.cardHeader}>
              <Text style={styles.cardLabel}>Current Price</Text>
              <View style={styles.lastUpdatedContainer}>
                <Ionicons name="time-outline" size={12} color="#888" />
                <Text style={styles.lastUpdated}>
                  <Text>Updated: </Text>
                  <Text>{priceData.lastUpdated.toString()}</Text>
                </Text>
              </View>
            </View>
            
            <View style={styles.priceContainer}>
              <View>
                <Text style={styles.priceValue}>
                  {priceData.current.toFixed(2).toString()}
                </Text>
                <Text style={styles.priceCurrency}>
                  <Text>{priceData.currency.toString()} </Text>
                  <Text>{priceData.unit.toString()}</Text>
                </Text>
              </View>
              
              <View style={styles.priceChangeTrend}>
                {renderPriceChangeIndicator(priceData.priceChange.daily)}
                <Text style={styles.todayText}>Today</Text>
              </View>
            </View>
          </View>

          {/* Enhanced Price Changes Card */}
          <View style={styles.priceChangesCard}>
            <Text style={styles.cardLabel}>Price Trends</Text>
            <View style={styles.changeRow}>
              <View style={styles.periodBadge}>
                <Text style={styles.periodText}>24h</Text>
              </View>
              <Text style={styles.changeLabel}>Daily Change</Text>
              {renderPriceChangeIndicator(priceData.priceChange.daily)}
            </View>
            <View style={styles.divider} />
            
            <View style={styles.changeRow}>
              <View style={[styles.periodBadge, { backgroundColor: '#e1f0e7' }]}>
                <Text style={styles.periodText}>7d</Text>
              </View>
              <Text style={styles.changeLabel}>Weekly Change</Text>
              {renderPriceChangeIndicator(priceData.priceChange.weekly)}
            </View>
            <View style={styles.divider} />
            
            <View style={styles.changeRow}>
              <View style={[styles.periodBadge, { backgroundColor: '#d0e7d9' }]}>
                <Text style={styles.periodText}>30d</Text>
              </View>
              <Text style={styles.changeLabel}>Monthly Change</Text>
              {renderPriceChangeIndicator(priceData.priceChange.monthly)}
            </View>
          </View>

          {/* Enhanced Price Chart Card */}
          <View style={styles.chartCard}>
            <View style={styles.chartHeader}>
              <Text style={styles.cardLabel}>Price Trend (2025)</Text>
              <View style={styles.chartLegend}>
                <View style={styles.legendItem}>
                  <View style={[styles.legendColor, { backgroundColor: '#2e8b57' }]} />
                  <Text style={styles.legendText}>Highest</Text>
                </View>
                <View style={styles.legendItem}>
                  <View style={[styles.legendColor, { backgroundColor: '#87c4a3' }]} />
                  <Text style={styles.legendText}>Standard</Text>
                </View>
              </View>
            </View>
            <SimpleBarChart 
              data={priceData.historicalData.datasets[0].data}
              labels={priceData.historicalData.labels}
              width={screenWidth - 60}
              height={220}
            />
          </View>

          {/* Enhanced Monthly Average Prices */}
          <View style={styles.monthlyAvgCard}>
            <Text style={styles.cardLabel}>Monthly Average Prices</Text>
            <View style={styles.monthlyHeader}>
              <Text style={styles.monthHeaderText}>Month</Text>
              <Text style={styles.priceHeaderText}>Price ({priceData.currency})</Text>
            </View>
            {priceData.monthlyAverages.map((item, index) => (
              <View key={index} style={styles.monthRow}>
                <Text style={styles.monthName}>{item.month}</Text>
                <Text style={styles.monthPrice}>
                  <Text>{item.price.toFixed(2).toString()}</Text>
                </Text>
              </View>
            ))}
          </View>

          {/* Enhanced Disclaimer */}
          <View style={styles.disclaimerCard}>
            <View style={styles.disclaimerHeader}>
              <Ionicons name="information-circle" size={16} color="#888" />
              <Text style={styles.disclaimerTitle}>Disclaimer</Text>
            </View>
            <Text style={styles.disclaimerText}>
              <Text>The prices shown are for informational purposes only. </Text>
              <Text>Actual market prices may vary based on location, quality, and market conditions. </Text>
              <Text>This data will be updated regularly once connected to the Commodities API.</Text>
            </Text>
          </View>
        </ScrollView>
      ) : (
        <View style={styles.loadingContainer}>
          <Ionicons name="alert-circle-outline" size={48} color="#888" />
          <Text style={styles.errorText}>Failed to load price data. Please try again later.</Text>          <TouchableOpacity 
            style={styles.retryButton}
            onPress={() => {
              setLoading(true);
              const fetchData = async () => {
                try {
                  const data = await getCurrentPalmOilPrice();
                  setPriceData(data as PriceData);
                  setLoading(false);
                } catch (error) {
                  console.error('Error fetching palm oil price data:', error);
                  setLoading(false);
                }
              };
              fetchData();
            }}
          >
            <Text style={styles.retryText}>Retry</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  headerGradient: {
    paddingTop: 60,
    paddingBottom: 20,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
  },
  backButton: {
    padding: 5,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
  },
  infoButton: {
    padding: 5,
  },
  placeholder: {
    width: 34,
    height: 34,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  contentContainer: {
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  scrollContent: {
    paddingBottom: 20,
  },
  priceCard: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 15,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  cardLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2e8b57',
  },
  lastUpdatedContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  lastUpdated: {
    marginLeft: 5,
    fontSize: 12,
    color: '#888',
  },
  priceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  priceValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2e8b57',
  },
  priceCurrency: {
    fontSize: 16,
    color: '#666',
  },
  priceChangeTrend: {
    alignItems: 'center',
  },
  todayText: {
    fontSize: 12,
    color: '#666',
    marginTop: 5,
  },
  priceChangesCard: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 15,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  changeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  periodBadge: {
    backgroundColor: '#f0f8f5',
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  periodText: {
    fontSize: 12,
    color: '#2e8b57',
  },
  changeLabel: {
    fontSize: 14,
    color: '#666',
  },
  divider: {
    height: 1,
    backgroundColor: '#eee',
    marginVertical: 10,
  },
  chartCard: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 15,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  chartHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  chartLegend: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: 10,
  },
  legendColor: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 5,
  },
  legendText: {
    fontSize: 12,
    color: '#666',
  },
  monthlyAvgCard: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 15,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  monthlyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  monthHeaderText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#666',
  },
  priceHeaderText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#666',
  },
  monthRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  monthName: {
    fontSize: 14,
    color: '#666',
  },
  monthPrice: {
    fontSize: 14,
    color: '#666',
  },
  disclaimerCard: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 15,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  disclaimerHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  disclaimerTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#666',
    marginLeft: 5,
  },
  disclaimerText: {
    fontSize: 12,
    color: '#666',
  },
  errorText: {
    fontSize: 16,
    color: '#666',
  },
  retryButton: {
    marginTop: 20,
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: '#2e8b57',
    borderRadius: 8,
  },
  retryText: {
    fontSize: 14,
    color: 'white',
    textAlign: 'center',
  },
});
