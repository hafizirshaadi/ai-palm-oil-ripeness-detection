import {
  Text,
  View,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { useAuth } from "../contexts/AuthContext";

export default function HomeScreen() {
  const router = useRouter();
  const { user, logout } = useAuth();

  const handleLogout = async () => {
    Alert.alert("Logout", "Are you sure you want to logout?", [
      {
        text: "Cancel",
        style: "cancel",
      },
      {
        text: "Logout",
        onPress: async () => {
          await logout();
          router.replace("/");
        },
      },
    ]);
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>PalmScan AI</Text>
        <TouchableOpacity onPress={handleLogout} style={styles.logoutButton}>
          <Ionicons name="log-out-outline" size={24} color="#2e8b57" />
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.contentContainer}>
        <View style={styles.welcomeContainer}>
          <Text style={styles.welcomeText}>
            Welcome, {user?.name || "Farmer"}!
          </Text>
          <Text style={styles.subtitleText}>
            Your smart palm fruit analysis tool
          </Text>
        </View>

        <View style={styles.cardsContainer}>
          <TouchableOpacity
            style={styles.card}
            onPress={() => router.push("/scan")}
          >
            <Ionicons name="scan-outline" size={32} color="#2e8b57" />
            <Text style={styles.cardTitle}>Scan Palm Fruit</Text>
            <Text style={styles.cardDescription}>
              Analyze palm fruit quality and ripeness with AI
            </Text>
          </TouchableOpacity>

          {/* <TouchableOpacity
            style={styles.card}
            onPress={() => router.push("/history")}
          >
            <Ionicons name="time-outline" size={32} color="#2e8b57" />
            <Text style={styles.cardTitle}>Scan History</Text>
            <Text style={styles.cardDescription}>
              View your previous palm fruit scan results
            </Text>
          </TouchableOpacity> */}

          <TouchableOpacity
            style={styles.card}
            onPress={() => router.push("/palm-oil-price")}
          >
            <Ionicons name="trending-up-outline" size={32} color="#2e8b57" />
            <Text style={styles.cardTitle}>Oil Palm Price</Text>
            <Text style={styles.cardDescription}>
              Check current prices and market trends for palm oil
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.card}
            onPress={() => router.push("/farm-mapping")}
          >
            <Ionicons name="map-outline" size={32} color="#2e8b57" />
            <Text style={styles.cardTitle}>Farm Mapping</Text>
            <Text style={styles.cardDescription}>
              Monitor and manage tree health across your plantation
            </Text>
          </TouchableOpacity>

          {/* <TouchableOpacity
            style={styles.card}
            onPress={() => router.push("/analytics")}
          >
            <Ionicons name="bar-chart-outline" size={32} color="#e4a84c" />
            <Text style={styles.cardTitle}>Analytics</Text>
            <Text style={styles.cardDescription}>
              Track fruit quality trends and harvest insights
            </Text>
          </TouchableOpacity> */}

            <TouchableOpacity
            style={styles.card}
            onPress={() => router.push("/education-resources")}
            >
            <Ionicons name="leaf-outline" size={32} color="#e4a84c" />
            <Text style={styles.cardTitle}>Education Resources</Text>
            <Text style={styles.cardDescription}>
              Learn best practices for cultivating, harvesting, and managing palm fruit
            </Text>
            </TouchableOpacity>

          {/* <TouchableOpacity
            style={styles.card}
            onPress={() => router.push("/marketplace")}
          >
            <Ionicons name="basket-outline" size={32} color="#e4a84c" />
            <Text style={styles.cardTitle}>Marketplace</Text>
            <Text style={styles.cardDescription}>
              Connect with buyers and get fair prices for your harvest
            </Text>
          </TouchableOpacity> */}

          <TouchableOpacity
            style={styles.card}
            onPress={() => router.push("/settings")}
          >
            <Ionicons name="settings-outline" size={32} color="#666" />
            <Text style={styles.cardTitle}>Settings</Text>
            <Text style={styles.cardDescription}>
              Customize your app preferences and scanner settings
            </Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f7f9fc",
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 20,
    backgroundColor: "white",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 3,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#2e8b57",
  },
  logoutButton: {
    padding: 5,
  },
  contentContainer: {
    flex: 1,
    padding: 20,
  },
  welcomeContainer: {
    marginBottom: 30,
  },
  welcomeText: {
    fontSize: 28,
    fontWeight: "bold",
    color: "#333",
  },
  subtitleText: {
    fontSize: 16,
    color: "#666",
    marginTop: 5,
  },
  cardsContainer: {
    gap: 16,
  },
  card: {
    backgroundColor: "white",
    borderRadius: 12,
    padding: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.08,
    shadowRadius: 2,
    elevation: 2,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "600",
    color: "#333",
    marginTop: 10,
    marginBottom: 6,
  },
  cardDescription: {
    fontSize: 14,
    color: "#666",
    lineHeight: 20,
  },
});
