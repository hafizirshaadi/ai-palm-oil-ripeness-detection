import React, { useState, useEffect, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Image,
  Alert,
  RefreshControl,
  ActivityIndicator,
  Dimensions,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { useAuth } from "../contexts/AuthContext";
import { WebView } from "react-native-webview";
import * as ImagePicker from "expo-image-picker";
import { 
  fetchPosts as fetchEducationPosts, 
  createPost, 
  deletePost, 
  likePost 
} from "../../utils/educationResourcesService";

// Type definitions
type Post = {
  _id: string;
  userId: string;
  userName: string;
  content: string;
  mediaUrl?: string;
  youtubeUrl?: string;
  likes: string[];
  createdAt: string;
};

export default function EducationResourcesScreen() {
  const router = useRouter();
  const { user } = useAuth();
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [showMyPosts, setShowMyPosts] = useState(false);
  const [newPostContent, setNewPostContent] = useState("");
  const [newPostYoutubeUrl, setNewPostYoutubeUrl] = useState("");
  const [image, setImage] = useState<string | null>(null);
  const [showNewPostForm, setShowNewPostForm] = useState(false);
  const fetchPosts = useCallback(async () => {
    try {
      setLoading(true);
      const data = await fetchEducationPosts();
      setPosts(data);
    } catch (error) {
      console.error("Error fetching posts:", error);
      Alert.alert("Error", "Could not load education resources");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchPosts();
  }, [fetchPosts]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchPosts();
  }, [fetchPosts]);
  const handleCreatePost = async () => {
    if (!newPostContent.trim()) {
      Alert.alert("Error", "Post content cannot be empty");
      return;
    }

    try {
      setLoading(true);      const postData = {
        userId: user?.id,
        userName: user?.name,
        content: newPostContent,
        youtubeUrl: newPostYoutubeUrl || undefined,
        mediaUrl: image || undefined,
      };

      await createPost(postData);

      setNewPostContent("");
      setNewPostYoutubeUrl("");
      setImage(null);
      setShowNewPostForm(false);
      await fetchPosts();
      Alert.alert("Success", "Post created successfully!");
    } catch (error) {
      console.error("Error creating post:", error);
      Alert.alert("Error", "Could not create post");
    } finally {
      setLoading(false);
    }
  };
  const handleDeletePost = async (postId: string) => {
    Alert.alert(
      "Delete Post",
      "Are you sure you want to delete this post?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          onPress: async () => {
            try {
              setLoading(true);
              await deletePost(postId);
              await fetchPosts();
              Alert.alert("Success", "Post deleted successfully!");
            } catch (error) {
              console.error("Error deleting post:", error);
              Alert.alert("Error", "Could not delete post");
            } finally {
              setLoading(false);
            }
          },
          style: "destructive",
        },
      ]
    );
  };
  const handleLikePost = async (postId: string) => {
    try {
      if (!user?.id) {
        Alert.alert("Error", "You need to be logged in to like posts");
        return;
      }
      await likePost(postId, user.id);
      await fetchPosts();
    } catch (error) {
      console.error("Error liking post:", error);
      Alert.alert("Error", "Could not like post");
    }
  };

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 1,
      });

      if (!result.canceled) {
        setImage(result.assets[0].uri);
      }
    } catch (error) {
      console.error("Error picking image:", error);
      Alert.alert("Error", "Could not pick image");
    }
  };

  const getYoutubeVideoId = (url: string) => {
    if (!url) return null;
    
    const regex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/;
    const match = url.match(regex);
    return match ? match[1] : null;
  };

  const renderPosts = () => {
    if (loading && !refreshing) {
      return (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2e8b57" />
        </View>
      );
    }

    if (posts.length === 0) {
      return (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>No posts found</Text>
        </View>
      );
    }    const filteredPosts = showMyPosts
      ? posts.filter((post) => post.userId === user?.id)
      : posts;

    if (filteredPosts.length === 0) {
      return (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>You haven&apos;t created any posts yet</Text>
        </View>
      );
    }

    return filteredPosts.map((post) => {
      const isPostOwner = post.userId === user?.id;
      const isLiked = post.likes.includes(user?.id || "");
      const youtubeVideoId = getYoutubeVideoId(post.youtubeUrl || "");

      return (
        <View key={post._id} style={styles.postCard}>
          <View style={styles.postHeader}>
            <Text style={styles.userName}>{post.userName}</Text>
            <Text style={styles.postDate}>
              {new Date(post.createdAt).toLocaleDateString()}
            </Text>
          </View>

          <Text style={styles.postContent}>{post.content}</Text>

          {post.mediaUrl && (
            <Image source={{ uri: post.mediaUrl }} style={styles.postImage} />
          )}

          {youtubeVideoId && (
            <View style={styles.youtubeContainer}>
              <WebView
                style={styles.youtubeVideo}
                javaScriptEnabled={true}
                domStorageEnabled={true}
                source={{
                  uri: `https://www.youtube.com/embed/${youtubeVideoId}`,
                }}
              />
            </View>
          )}

          <View style={styles.postActions}>
            <TouchableOpacity
              style={styles.likeButton}
              onPress={() => handleLikePost(post._id)}
            >
              <Ionicons
                name={isLiked ? "heart" : "heart-outline"}
                size={24}
                color={isLiked ? "#e74c3c" : "#666"}
              />
              <Text style={styles.likeCount}>{post.likes.length}</Text>
            </TouchableOpacity>

            {isPostOwner && (
              <TouchableOpacity
                style={styles.deleteButton}
                onPress={() => handleDeletePost(post._id)}
              >
                <Ionicons name="trash-outline" size={24} color="#666" />
              </TouchableOpacity>
            )}
          </View>
        </View>
      );
    });
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => router.back()}
        >
          <Ionicons name="chevron-back" size={24} color="#2e8b57" />
        </TouchableOpacity>
        <Text style={styles.title}>Education Resources</Text>
        <View style={{ width: 24 }} />
      </View>

      <View style={styles.tabContainer}>
        <TouchableOpacity
          style={[
            styles.tab,
            !showMyPosts && styles.activeTab,
          ]}
          onPress={() => setShowMyPosts(false)}
        >
          <Text
            style={[
              styles.tabText,
              !showMyPosts && styles.activeTabText,
            ]}
          >
            All Posts
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[
            styles.tab,
            showMyPosts && styles.activeTab,
          ]}
          onPress={() => setShowMyPosts(true)}
        >
          <Text
            style={[
              styles.tabText,
              showMyPosts && styles.activeTabText,
            ]}
          >
            My Posts
          </Text>
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.contentContainer}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {showNewPostForm && (
          <View style={styles.newPostForm}>
            <Text style={styles.inputLabel}>Post Content</Text>
            <TextInput
              style={styles.textInput}
              placeholder="Share your knowledge..."
              value={newPostContent}
              onChangeText={setNewPostContent}
              multiline
            />

            <Text style={styles.inputLabel}>YouTube URL (optional)</Text>
            <TextInput
              style={styles.textInput}
              placeholder="YouTube URL (optional)"
              value={newPostYoutubeUrl}
              onChangeText={setNewPostYoutubeUrl}
            />

            <Text style={styles.inputLabel}>Add Image (optional)</Text>
            <View style={styles.newPostActions}>
              <TouchableOpacity
                style={styles.imageButton}
                onPress={pickImage}
              >
                <Ionicons name="image-outline" size={24} color="#2e8b57" />
                <Text style={styles.buttonText}>Add Image</Text>
              </TouchableOpacity>

              {image && (
                <View style={styles.imagePreviewContainer}>
                  <Image source={{ uri: image }} style={styles.imagePreview} />
                  <TouchableOpacity
                    style={styles.removeImageButton}
                    onPress={() => setImage(null)}
                  >
                    <Ionicons name="close-circle" size={24} color="#e74c3c" />
                  </TouchableOpacity>
                </View>
              )}

              <View style={styles.formButtons}>
                <TouchableOpacity
                  style={[styles.button, styles.cancelButton]}
                  onPress={() => {
                    setShowNewPostForm(false);
                    setNewPostContent("");
                    setNewPostYoutubeUrl("");
                    setImage(null);
                  }}
                >
                  <Text style={styles.cancelButtonText}>Cancel</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={[
                    styles.button,
                    styles.postButton,
                    !newPostContent.trim() && styles.disabledButton,
                  ]}
                  onPress={handleCreatePost}
                  disabled={!newPostContent.trim()}
                >
                  <Text style={styles.postButtonText}>Post</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        )}

        {renderPosts()}
      </ScrollView>

      {!showNewPostForm && (
        <TouchableOpacity
          style={styles.fab}
          onPress={() => setShowNewPostForm(true)}
        >
          <Ionicons name="add" size={30} color="#fff" />
        </TouchableOpacity>
      )}
    </View>  );
}

// Get screen width for responsive styling
const { width: screenWidth } = Dimensions.get("window");

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
  backButton: {
    padding: 5,
  },
  title: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#2e8b57",
  },
  tabContainer: {
    flexDirection: "row",
    backgroundColor: "white",
    borderBottomWidth: 1,
    borderBottomColor: "#e1e1e1",
  },
  tab: {
    flex: 1,
    paddingVertical: 15,
    alignItems: "center",
  },
  activeTab: {
    borderBottomWidth: 2,
    borderBottomColor: "#2e8b57",
  },
  tabText: {
    fontSize: 16,
    color: "#666",
  },
  activeTabText: {
    color: "#2e8b57",
    fontWeight: "600",
  },
  contentContainer: {
    flex: 1,
    padding: 16,
  },
  postCard: {
    backgroundColor: "white",
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.08,
    shadowRadius: 2,
    elevation: 2,
  },
  postHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 10,
  },
  userName: {
    fontSize: 16,
    fontWeight: "600",
    color: "#333",
  },
  postDate: {
    fontSize: 12,
    color: "#999",
  },
  postContent: {
    fontSize: 16,
    color: "#333",
    marginBottom: 12,
    lineHeight: 24,
  },  postImage: {
    width: "100%",
    height: screenWidth * 0.6, // Responsive height based on screen width
    borderRadius: 8,
    marginBottom: 12,
    resizeMode: "cover",
  },youtubeContainer: {
    width: "100%",
    height: screenWidth * 0.56, // 16:9 aspect ratio
    marginBottom: 12,
    borderRadius: 8,
    overflow: "hidden",
  },
  youtubeVideo: {
    flex: 1,
  },
  postActions: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  likeButton: {
    flexDirection: "row",
    alignItems: "center",
  },
  likeCount: {
    marginLeft: 5,
    fontSize: 16,
    color: "#666",
  },
  deleteButton: {
    padding: 5,
  },
  fab: {
    position: "absolute",
    right: 20,
    bottom: 20,
    backgroundColor: "#2e8b57",
    width: 60,
    height: 60,
    borderRadius: 30,
    justifyContent: "center",
    alignItems: "center",
    elevation: 5,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 3,
  },
  newPostForm: {
    backgroundColor: "white",
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.08,
    shadowRadius: 2,
    elevation: 2,
  },
  inputLabel: {
    fontSize: 14,
    color: "#333",
    marginBottom: 4,
  },
  textInput: {
    borderWidth: 1,
    borderColor: "#ddd",
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    marginBottom: 12,
    backgroundColor: "#f9f9f9",
  },
  newPostActions: {
    marginTop: 8,
  },
  imageButton: {
    flexDirection: "row",
    alignItems: "center",
    padding: 8,
  },
  buttonText: {
    marginLeft: 8,
    fontSize: 16,
    color: "#2e8b57",
  },
  imagePreviewContainer: {
    position: "relative",
    marginTop: 10,
    marginBottom: 16,
  },  imagePreview: {
    width: "100%",
    height: screenWidth * 0.6,
    borderRadius: 8,
    resizeMode: "cover",
  },
  removeImageButton: {
    position: "absolute",
    top: 10,
    right: 10,
    backgroundColor: "rgba(255,255,255,0.8)",
    borderRadius: 15,
  },
  formButtons: {
    flexDirection: "row",
    justifyContent: "flex-end",
    marginTop: 12,
  },
  button: {
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginLeft: 10,
  },
  postButton: {
    backgroundColor: "#2e8b57",
  },
  postButtonText: { // New style for the Post button text
    fontSize: 16,
    color: "#fff",
    fontWeight: "bold", // Optional: for better emphasis
  },
  cancelButton: {
    backgroundColor: "transparent",
    borderWidth: 1,
    borderColor: "#ddd",
  },
  cancelButtonText: {
    color: "#666",
    fontSize: 16,
  },
  disabledButton: {
    backgroundColor: "#ccc",
    opacity: 0.7,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  emptyText: {
    fontSize: 16,
    color: "#666",
    textAlign: "center",
  },
});
