/**
 * Education Resources Service
 * This service handles API calls to the backend for education resources functionality
 */

import { Alert } from 'react-native';

// Update this to your actual backend URL
const API_URL = 'http://172.20.10.4:3000/api';

/**
 * Get all posts from the education resources
 * @returns {Promise<Array>} Array of post objects
 */
export const fetchPosts = async () => {
  try {
    const response = await fetch(`${API_URL}/education-resources`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching posts:', error);
    throw error;
  }
};

/**
 * Create a new post
 * @param {Object} postData - The post data to create
 * @returns {Promise<Object>} The created post object
 */
export const createPost = async (postData) => {
  try {
    const response = await fetch(`${API_URL}/education-resources`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(postData),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error creating post:', error);
    throw error;
  }
};

/**
 * Delete a post
 * @param {string} postId - The ID of the post to delete
 * @returns {Promise<boolean>} Success status
 */
export const deletePost = async (postId) => {
  try {
    const response = await fetch(`${API_URL}/education-resources/${postId}`, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    return true;
  } catch (error) {
    console.error('Error deleting post:', error);
    throw error;
  }
};

/**
 * Like a post
 * @param {string} postId - The ID of the post to like
 * @param {string} userId - The ID of the user who is liking the post
 * @returns {Promise<Object>} The updated post object
 */
export const likePost = async (postId, userId) => {
  try {
    const response = await fetch(`${API_URL}/education-resources/${postId}/like`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ userId }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error liking post:', error);
    throw error;
  }
};
