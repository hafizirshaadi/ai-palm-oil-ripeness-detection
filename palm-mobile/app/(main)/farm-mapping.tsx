import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  TextInput,
  Alert,
  Modal,
  Switch,
  Platform,
  KeyboardAvoidingView,
  TouchableWithoutFeedback,
  Keyboard,
  Dimensions,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';

// Mock SVG components until react-native-svg is properly installed
const Svg = ({ children, width, height, ...props }) => (
  <View style={{ width, height, ...props }}>{children}</View>
);

// Custom TreeIcon component to replace Circle
const TreeIcon = ({ cx, cy, size, status, onPress, treeId }) => (
  <TouchableOpacity
    onPress={onPress}
    style={{
      position: 'absolute',
      left: cx - size/2,
      top: cy - size/2,
      width: size,
      height: size,
      alignItems: 'center',
      justifyContent: 'center',
    }}
  >
    <Text style={{ fontSize: size * 0.9 }}>🌴</Text>
    <View style={{
      position: 'absolute',
      bottom: -10,
      backgroundColor: status === 'healthy' ? 'rgba(40, 167, 69, 0.8)' : 'rgba(220, 53, 69, 0.8)',
      paddingHorizontal: 4,
      paddingVertical: 2,
      borderRadius: 10,
    }}>
      <Text style={{ 
        color: 'white', 
        fontSize: 10, 
        fontWeight: 'bold' 
      }}>
        {treeId}
      </Text>
    </View>
  </TouchableOpacity>
);

const Line = ({ x1, y1, x2, y2, stroke, strokeWidth }) => (
  <View
    style={{
      position: 'absolute',
      backgroundColor: stroke,
      height: strokeWidth,
      width: x2 - x1,
      left: x1,
      top: y1,
    }}
  />
);

const SvgText = ({ x, y, children, fill, fontSize, fontWeight, textAnchor }) => (
  <Text
    style={{
      position: 'absolute',
      color: fill,
      fontSize,
      fontWeight,
      textAlign: textAnchor === 'middle' ? 'center' : 'left',
      left: x - (textAnchor === 'middle' ? 10 : 0),
      top: y - 10,
      width: 20,
    }}
  >
    {children}
  </Text>
);

// Define types for our data
interface Tree {
  _id?: string;
  id?: number;
  x: number;
  y: number;
  status: 'healthy' | 'sick';
  description: string;
  timestamp: string;
}

interface TreeHistory {
  _id?: string;
  id?: number;
  treeId: string | number;
  status: 'healthy' | 'sick';
  description: string;
  timestamp: string;
}

// Service function to handle API calls
const apiBaseUrl = 'http://172.20.10.4:3000/api';;

const farmMappingService = {  // Get all trees
  fetchTrees: async (): Promise<Tree[]> => {
    try {
      console.log("Fetching trees from:", `${apiBaseUrl}/trees`);
      // For production, use the actual API URL
      const response = await fetch(`${apiBaseUrl}/trees`);
      
      if (!response.ok) {
        console.log("HTTP error status:", response.status);
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Received data:", data);
      return data;
    } catch (error) {
      console.error('Error fetching trees:', error);
      throw error;
    }
  },  // Get tree history
  fetchTreeHistory: async (treeId: string | number): Promise<TreeHistory[]> => {
    try {
      console.log("Fetching tree history from:", `${apiBaseUrl}/trees/${treeId}/history`);
      const response = await fetch(`${apiBaseUrl}/trees/${treeId}/history`);
      
      if (!response.ok) {
        console.log("HTTP error status:", response.status);
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Received history data:", data);
      return data;
    } catch (error) {
      console.error('Error fetching tree history:', error);
      throw error;
    }
  },  // Add new tree
  addTree: async (tree: Tree): Promise<Tree> => {
    try {
      console.log("Adding new tree to:", `${apiBaseUrl}/trees`, "with data:", tree);
      const response = await fetch(`${apiBaseUrl}/trees`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(tree),
      });
      
      if (!response.ok) {
        console.log("HTTP error status:", response.status);
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Created tree:", data);
      return data;
    } catch (error) {
      console.error('Error adding tree:', error);
      throw error;
    }
  },  // Update tree
  updateTree: async (treeId: string | number, treeData: Partial<Tree>): Promise<Tree> => {
    try {
      console.log("Updating tree at:", `${apiBaseUrl}/trees/${treeId}`, "with data:", treeData);
      const response = await fetch(`${apiBaseUrl}/trees/${treeId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(treeData),
      });
      
      if (!response.ok) {
        console.log("HTTP error status:", response.status);
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Updated tree:", data);
      return data;
    } catch (error) {
      console.error('Error updating tree:', error);
      throw error;
    }
  },  // Delete tree
  deleteTree: async (treeId: string | number): Promise<boolean> => {
    try {
      console.log("Deleting tree at:", `${apiBaseUrl}/trees/${treeId}`);
      const response = await fetch(`${apiBaseUrl}/trees/${treeId}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        console.log("HTTP error status:", response.status);
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      console.log("Tree deleted successfully");
      return true;
    } catch (error) {
      console.error('Error deleting tree:', error);
      throw error;
    }
  }
};

// Mock data for trees for fallback during development
const initialTrees: Tree[] = [
  { id: 1, x: 50, y: 50, status: 'healthy', description: 'Good growth', timestamp: new Date().toISOString() },
  { id: 2, x: 150, y: 100, status: 'sick', description: 'Yellow leaves', timestamp: new Date().toISOString() },
  { id: 3, x: 250, y: 170, status: 'healthy', description: 'Recently planted', timestamp: new Date().toISOString() },
  { id: 4, x: 100, y: 200, status: 'sick', description: 'Fungal infection', timestamp: new Date().toISOString() },
  { id: 5, x: 300, y: 80, status: 'healthy', description: 'Good yield', timestamp: new Date().toISOString() },
];

export default function FarmMappingScreen() {  
  const router = useRouter();
  const [trees, setTrees] = useState<Tree[]>(initialTrees);
  const [selectedTree, setSelectedTree] = useState<Tree | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [addMode, setAddMode] = useState(false);
  const [newTreeCoords, setNewTreeCoords] = useState({ x: 0, y: 0 });
  const [newDescription, setNewDescription] = useState("");
  const [isHealthy, setIsHealthy] = useState(true);
  const [treeHistory, setTreeHistory] = useState<TreeHistory[]>([]);
  const [historyModalVisible, setHistoryModalVisible] = useState(false);
  const graphRef = useRef(null);

  const dismissKeyboard = () => {
    Keyboard.dismiss();
  };

  // Fetch all trees on component mount
  useEffect(() => {
    fetchTrees();
  }, []);

  const fetchTrees = async () => {
    try {
      const trees = await farmMappingService.fetchTrees();
      setTrees(trees);
      console.log("Fetched trees from API");
    } catch (error) {
      console.error('Error fetching trees:', error);
      Alert.alert('Error', 'Failed to load farm mapping data');
      setTrees(initialTrees);
    }
  };

  const fetchTreeHistory = async (treeId: string | number) => {
    try {
      if (!treeId) {
        console.error('Tree ID is missing');
        setTreeHistory([]);
        return;
      }
      
      console.log("Fetching tree history from:", `${apiBaseUrl}/trees/${treeId}/history`);
      const response = await fetch(`${apiBaseUrl}/trees/${treeId}/history`);
      
      if (!response.ok) {
        console.log("HTTP error status:", response.status);
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Received history data:", data);
      setTreeHistory(data);
    } catch (error) {
      console.error('Error fetching tree history:', error);
      Alert.alert('Error', 'Failed to load tree history');
      
      // Fallback to mock history for development purposes only
      const mockHistory: TreeHistory[] = [
        { 
          id: 1, 
          treeId, 
          status: 'healthy', 
          description: 'Planted', 
          timestamp: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString() 
        },
        { 
          id: 2, 
          treeId, 
          status: selectedTree?.status === 'sick' ? 'healthy' : 'sick', 
          description: selectedTree?.status === 'sick' ? 'Started showing symptoms' : 'Treated with fungicide', 
          timestamp: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString() 
        },
        { 
          id: 3, 
          treeId, 
          status: selectedTree?.status || 'healthy', 
          description: selectedTree?.description || '', 
          timestamp: new Date().toISOString() 
        }
      ];
      setTreeHistory(mockHistory);
    }
  };

  const handleTreePress = (tree: Tree) => {
    dismissKeyboard();
    setSelectedTree(tree);
    setIsHealthy(tree.status === 'healthy');
    setNewDescription(tree.description);
    setModalVisible(true);
    const treeId = tree._id || tree.id;
    if (treeId) {
      fetchTreeHistory(treeId);
    } else {
      console.warn('Tree has no ID, cannot fetch history');
    }
  };

  const handleGraphPress = (event: any) => {
    dismissKeyboard();
    if (addMode) {
      const { locationX, locationY } = event.nativeEvent;
      setNewTreeCoords({ x: locationX, y: locationY });
      setModalVisible(true);
      setSelectedTree(null);
      setNewDescription('');
      setIsHealthy(true);
    }
  };

  const handleAddTree = async () => {
    try {
      const newTree: Tree = {
        id: Date.now(),
        x: newTreeCoords.x,
        y: newTreeCoords.y,
        status: isHealthy ? 'healthy' : 'sick',
        description: newDescription,
        timestamp: new Date().toISOString()
      };

      const savedTree = await farmMappingService.addTree(newTree);
      setTrees([...trees, savedTree || newTree]);
      setModalVisible(false);
      setAddMode(false);
      Alert.alert('Success', 'New tree added successfully');
    } catch (error) {
      console.error('Error adding tree:', error);
      Alert.alert('Error', 'Failed to add new tree');
    }
  };

  const handleUpdateTree = async () => {
    try {
      if (!selectedTree) return;
      const treeId = selectedTree._id || selectedTree.id;
      if (!treeId) {
        Alert.alert('Error', 'Tree ID is missing');
        return;
      }
      const updatedTreeData: Partial<Tree> = {
        status: isHealthy ? 'healthy' : 'sick',
        description: newDescription,
        timestamp: new Date().toISOString()
      };

      let updatedTree: Tree | null = null;
      try {
        updatedTree = await farmMappingService.updateTree(treeId, updatedTreeData);
      } catch (error) {
        console.error('Error updating tree on API:', error);
      }
      const updatedTrees = trees.map(tree => 
        (tree._id === treeId || tree.id === treeId) 
          ? (updatedTree || { ...tree, ...updatedTreeData }) 
          : tree
      );
      setTrees(updatedTrees as Tree[]);
      setModalVisible(false);
      Alert.alert('Success', 'Tree updated successfully');
    } catch (error) {
      console.error('Error updating tree:', error);
      Alert.alert('Error', 'Failed to update tree');
    }
  };

  const handleDeleteTree = async () => {
    try {
      if (!selectedTree) return;
      const treeId = selectedTree._id || selectedTree.id;
      if (!treeId) {
        Alert.alert('Error', 'Tree ID is missing');
        return;
      }
      Alert.alert(
        'Confirm Deletion',
        'Are you sure you want to delete this tree?',
        [
          {
            text: 'Cancel',
            style: 'cancel',
          },
          {
            text: 'Delete',
            onPress: async () => {
              try {
                await farmMappingService.deleteTree(treeId);
                const filteredTrees = trees.filter(tree => 
                  tree._id !== treeId && tree.id !== treeId
                );
                setTrees(filteredTrees);
                setModalVisible(false);
                Alert.alert('Success', 'Tree deleted successfully');
              } catch (error) {
                console.error('Error deleting tree:', error);
                Alert.alert('Error', 'Failed to delete tree');
              }
            },
            style: 'destructive',
          },
        ]
      );
    } catch (error) {
      console.error('Error in delete operation:', error);
      Alert.alert('Error', 'An unexpected error occurred');
    }
  };

  const handleViewHistory = () => {
    if (!selectedTree) {
      Alert.alert('Error', 'No tree selected');
      return;
    }

    const treeId = selectedTree._id || selectedTree.id;
    if (!treeId) {
      Alert.alert('Error', 'Tree ID is missing');
      return;
    }

    // fetch latest history
    fetchTreeHistory(treeId);

    // close the details modal so history modal can appear on top
    setModalVisible(false);
    setHistoryModalVisible(true);
  };

  const formatDate = (dateString: string) => {
    const options: Intl.DateTimeFormatOptions = { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric', 
      hour: '2-digit', 
      minute: '2-digit' 
    };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#2e8b57" />
        </TouchableOpacity>
        <Text style={styles.title}>Farm Mapping</Text>
        <TouchableOpacity 
          onPress={() => setAddMode(!addMode)}
          style={[styles.addButton, addMode && styles.addButtonActive]}
        >
          <Ionicons name={addMode ? "close" : "add"} size={24} color={addMode ? "#fff" : "#2e8b57"} />
        </TouchableOpacity>
      </View>

      <View style={styles.infoBar}>
        <Text style={styles.infoText}>
          {addMode 
            ? 'Tap on the map to add a new tree' 
            : 'Tap on a tree to view or update its status'}
        </Text>
      </View>

      <View style={styles.legendContainer}>
        <View style={styles.legendItem}>
          <Text style={{ fontSize: 18 }}>🌴</Text>
          <View style={{
            backgroundColor: 'rgba(40, 167, 69, 0.8)',
            paddingHorizontal: 4,
            paddingVertical: 1,
            borderRadius: 4,
            marginLeft: 4,
          }}>
            <Text style={[styles.legendText, { color: 'white', marginLeft: 0 }]}>Healthy</Text>
          </View>
        </View>
        <View style={styles.legendItem}>
          <Text style={{ fontSize: 18 }}>🌴</Text>
          <View style={{
            backgroundColor: 'rgba(220, 53, 69, 0.8)',
            paddingHorizontal: 4,
            paddingVertical: 1,
            borderRadius: 4,
            marginLeft: 4,
          }}>
            <Text style={[styles.legendText, { color: 'white', marginLeft: 0 }]}>Sick</Text>
          </View>
        </View>
      </View>

      <View 
        style={styles.graphContainer}
        ref={graphRef}
        onTouchEnd={handleGraphPress}
      >
        <LinearGradient
          colors={['#e8f5e9', '#c8e6c9', '#a5d6a7']}
          style={StyleSheet.absoluteFillObject}
        />
        <Svg height="100%" width="100%">
          {Array.from({ length: 11 }).map((_, i) => (
            <Line
              key={`horizontal-${i}`}
              x1="0"
              y1={i * 40}
              x2="100%"
              y2={i * 40}
              stroke="rgba(100, 150, 100, 0.3)"
              strokeWidth="1"
            />
          ))}
          {Array.from({ length: 11 }).map((_, i) => (
            <Line
              key={`vertical-${i}`}
              x1={i * 40}
              y1="0"
              x2={i * 40}
              y2="100%"
              stroke="rgba(100, 150, 100, 0.3)"
              strokeWidth="1"
            />
          ))}
          {trees.map((tree) => (
            <TreeIcon
              key={tree._id || tree.id}
              cx={tree.x}
              cy={tree.y}
              size={30}
              status={tree.status}
              treeId={tree._id ? tree._id.toString().slice(-3) : tree.id}
              onPress={() => handleTreePress(tree)}
            />
          ))}
        </Svg>
      </View>

      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => {
          setModalVisible(false);
          if (addMode) setAddMode(false);
        }}
      >
        <TouchableWithoutFeedback onPress={dismissKeyboard}>
          <KeyboardAvoidingView 
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            style={styles.centeredView}
          >
            <TouchableWithoutFeedback onPress={dismissKeyboard}>
              <View style={styles.modalView}>
                <Text style={styles.modalTitle}>
                  {selectedTree ? `Tree #${selectedTree._id ? selectedTree._id.toString().slice(-3) : selectedTree.id}` : 'Add New Tree'}
                </Text>

                <View style={styles.formGroup}>
                  <Text style={styles.label}>Status:</Text>
                  <View style={styles.switchContainer}>
                    <Text style={[styles.switchLabel, !isHealthy && styles.activeLabel]}>Sick</Text>
                    <Switch
                      trackColor={{ false: '#ffcccb', true: '#c8e6c9' }}
                      thumbColor={isHealthy ? '#28a745' : '#dc3545'}
                      ios_backgroundColor="#ffcccb"
                      onValueChange={setIsHealthy}
                      value={isHealthy}
                    />
                    <Text style={[styles.switchLabel, isHealthy && styles.activeLabel]}>Healthy</Text>
                  </View>
                </View>

                <View style={styles.formGroup}>
                  <Text style={styles.label}>Description:</Text>
                  <TextInput
                    style={styles.input}
                    value={newDescription}
                    onChangeText={setNewDescription}
                    placeholder="Enter tree description or notes"
                    multiline
                    returnKeyType="done"
                    blurOnSubmit={true}
                    onSubmitEditing={dismissKeyboard}
                  />
                </View>

                {selectedTree && (
                  <TouchableOpacity 
                    style={styles.historyButton}
                    onPress={handleViewHistory}
                  >
                    <Ionicons name="time-outline" size={18} color="#2e8b57" />
                    <Text style={styles.historyButtonText}>View History</Text>
                  </TouchableOpacity>
                )}

                <View style={styles.buttonRow}>
                  <TouchableOpacity
                    style={[styles.button, styles.cancelButton]}
                    onPress={() => {
                      dismissKeyboard();
                      setModalVisible(false);
                      if (addMode) setAddMode(false);
                    }}
                  >
                    <Text style={styles.buttonText}>Cancel</Text>
                  </TouchableOpacity>

                  {selectedTree ? (
                    <>
                      <TouchableOpacity
                        style={[styles.button, styles.deleteButton]}
                        onPress={handleDeleteTree}
                      >
                        <Text style={styles.buttonText}>Delete</Text>
                      </TouchableOpacity>
                      <TouchableOpacity
                        style={[styles.button, styles.saveButton]}
                        onPress={handleUpdateTree}
                      >
                        <Text style={styles.buttonText}>Update</Text>
                      </TouchableOpacity>
                    </>
                  ) : (
                    <TouchableOpacity
                      style={[styles.button, styles.saveButton]}
                      onPress={handleAddTree}
                    >
                      <Text style={styles.buttonText}>Add Tree</Text>
                    </TouchableOpacity>
                  )}
                </View>
              </View>
            </TouchableWithoutFeedback>
          </KeyboardAvoidingView>
        </TouchableWithoutFeedback>
      </Modal>

      <Modal
        animationType="slide"
        transparent={true}
        visible={historyModalVisible}
        onRequestClose={() => setHistoryModalVisible(false)}
      >
        <View style={styles.centeredView}>
          <View style={[styles.modalView, { maxHeight: '80%' }]}>
            <Text style={styles.modalTitle}>
              {selectedTree ? `Tree #${selectedTree._id ? selectedTree._id.toString().slice(-3) : selectedTree.id} History` : 'Tree History'}
            </Text>
            
            {treeHistory.length > 0 ? (
              <ScrollView style={styles.historyScrollView}>
                {treeHistory.map((entry, index) => (
                  <View key={entry._id || index} style={styles.historyItem}>
                    <View style={styles.historyHeader}>
                      <View style={[
                        styles.statusDot, 
                        { backgroundColor: entry.status === 'healthy' ? '#28a745' : '#dc3545' }
                      ]} />
                      <Text style={styles.historyDate}>{formatDate(entry.timestamp)}</Text>
                    </View>
                    <Text style={styles.historyStatus}>
                      Status: {entry.status === 'healthy' ? 'Healthy' : 'Sick'}
                    </Text>
                    <Text style={styles.historyDescription}>{entry.description}</Text>
                  </View>
                ))}
              </ScrollView>
            ) : (
              <View style={styles.emptyHistoryContainer}>
                <Ionicons name="time-outline" size={40} color="#ccc" />
                <Text style={styles.emptyHistoryText}>No history records found</Text>
              </View>
            )}
            
            <TouchableOpacity
              style={[styles.button, styles.closeButton]}
              onPress={() => setHistoryModalVisible(false)}
            >
              <Text style={styles.buttonText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f0f8ff",
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingBottom: 20,
    backgroundColor: "#f8fafc",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 3,
  },
  backButton: {
    padding: 5,
  },
  title: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#2e8b57",
  },
  addButton: {
    padding: 5,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#2e8b57',
    width: 36,
    height: 36,
    justifyContent: 'center',
    alignItems: 'center',
  },
  addButtonActive: {
    backgroundColor: '#2e8b57',
    borderColor: '#2e8b57',
  },
  infoBar: {
    backgroundColor: '#f0f8ff',
    padding: 10,
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#e1e1e1',
  },
  infoText: {
    color: '#666',
    fontSize: 14,
  },
  legendContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    padding: 10,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e1e1e1',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: 10,
  },
  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 6,
  },
  legendText: {
    fontSize: 14,
    color: '#666',
    marginLeft: 6,
  },
  graphContainer: {
    flex: 1,
    margin: 2,
    borderRadius: 10,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
    overflow: 'hidden',
  },
  centeredView: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  modalView: {
    width: '90%',
    backgroundColor: 'white',
    borderRadius: 20,
    padding: 25,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5,
  },
  modalTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 20,
    textAlign: 'center',
  },
  formGroup: {
    marginBottom: 20,
  },
  label: {
    fontSize: 16,
    color: '#666',
    marginBottom: 8,
  },
  switchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  switchLabel: {
    fontSize: 16,
    color: '#999',
    marginHorizontal: 10,
  },
  activeLabel: {
    color: '#333',
    fontWeight: '600',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    color: '#333',
    backgroundColor: '#f9f9f9',
    minHeight: 100,
    textAlignVertical: 'top',
  },
  historyButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    padding: 8,
    backgroundColor: '#f0f8ff',
    borderRadius: 8,
  },
  historyButtonText: {
    color: '#2e8b57',
    fontSize: 16,
    marginLeft: 8,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  button: {
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  saveButton: {
    backgroundColor: '#2e8b57',
    flex: 1,
    marginLeft: 8,
  },
  deleteButton: {
    backgroundColor: '#dc3545',
    flex: 1,
    marginHorizontal: 8,
  },
  cancelButton: {
    backgroundColor: '#f0f0f0',
    flex: 1,
    marginRight: 8,
  },
  closeButton: {
    backgroundColor: '#f0f0f0',
    alignSelf: 'center',
    paddingHorizontal: 40,
    marginTop: 10,
  },
  buttonText: {
    color: '#333',
    fontSize: 16,
    fontWeight: '600',
  },
  historyScrollView: {
    maxHeight: 400,
    marginBottom: 20,
    width: '100%',
  },
  historyItem: {
    backgroundColor: '#f9f9f9',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
    borderLeftWidth: 3,
    borderLeftColor: '#ddd',
  },
  historyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  statusDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 10,
  },
  historyDate: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  historyStatus: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 6,
  },
  historyDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
  emptyHistoryContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
  },
  emptyHistoryText: {
    marginTop: 10,
    fontSize: 16,
    color: '#888',
    textAlign: 'center',
  },
});
