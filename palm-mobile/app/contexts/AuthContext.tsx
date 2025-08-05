import React, { createContext, useState, useContext, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Types
type User = {
  id: string;
  name: string;
  email: string;
  isAdmin?: boolean;
  status?: string;
};

type AuthContextType = {
  user: User | null;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<{ success: boolean; message: string }>;
  register: (name: string, email: string, password: string) => Promise<{ success: boolean; message: string }>;
  logout: () => Promise<void>;
  updateProfile: (name: string) => Promise<{ success: boolean; message: string }>;
  changePassword: (currentPassword: string, newPassword: string) => Promise<{ success: boolean; message: string }>;
  deleteAccount: () => Promise<{ success: boolean; message: string }>;
  fetchWithAuth: (url: string, options?: RequestInit) => Promise<Response>;
};

// Create the context
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// API URL - Change this to your actual backend URL
// Use your computer's actual IP address instead of localhost
const API_URL = 'http://172.20.10.4:3000/api'; // Local backend URL

// Auth Provider component
export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in when app loads
    loadUser();
  }, []);

  const loadUser = async () => {
    try {
      const userJSON = await AsyncStorage.getItem('user');
      if (userJSON) {
        setUser(JSON.parse(userJSON));
      }
    } catch (error) {
      console.error('Failed to load user from storage', error);
    } finally {
      setIsLoading(false);
    }
  };
  // Helper function to make authenticated requests
  const fetchWithAuth = async (url: string, options: RequestInit = {}) => {
    const headers = {
      'Content-Type': 'application/json',
      ...(user?.id ? { 'x-user-id': user.id } : {}),
      ...(options.headers || {})
    };

    return fetch(url.startsWith('http') ? url : `${API_URL}${url}`, {
      ...options,
      headers
    });
  };

  const login = async (email: string, password: string) => {
    try {
      // Use the new mobile login endpoint
      const response = await fetch(`${API_URL}/auth/mobile/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (!response.ok) {
        return { success: false, message: data.message || 'Login failed' };
      }

      if (!data.success) {
        return { success: false, message: data.message || 'Login failed' };
      }

      // Store user data
      await AsyncStorage.setItem('user', JSON.stringify(data.user));
      setUser(data.user);

      return { success: true, message: 'Login successful' };
    } catch (error) {
      console.error('Login error:', error);
      return { success: false, message: 'Network error, please try again' };
    }
  };

  const register = async (name: string, email: string, password: string) => {
    try {
      const response = await fetch(`${API_URL}/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name, email, password }),
      });

      const data = await response.json();

      if (!response.ok) {
        return { success: false, message: data.message || 'Registration failed' };
      }

      return { success: true, message: 'Registration successful' };
    } catch (error) {
      console.error('Registration error:', error);
      return { success: false, message: 'Network error, please try again' };
    }
  };
  const logout = async () => {
    try {
      await AsyncStorage.removeItem('user');
      await AsyncStorage.removeItem('token');
      setUser(null);
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const updateProfile = async (name: string) => {
    try {
      if (!user) {
        return { success: false, message: 'You must be logged in' };
      }

      const response = await fetchWithAuth('/user/profile', {
        method: 'PUT',
        body: JSON.stringify({ name, email: user.email }),
      });

      const data = await response.json();

      if (!response.ok) {
        return { success: false, message: data.message || 'Profile update failed' };
      }

      // Update the user in state and storage
      // Make sure to preserve the status and other fields
      const updatedUser = { 
        ...user, 
        name,
        status: data.user?.status || user.status // Ensure status is updated if returned from API
      };
      setUser(updatedUser);
      await AsyncStorage.setItem('user', JSON.stringify(updatedUser));

      return { success: true, message: 'Profile updated successfully' };
    } catch (error) {
      console.error('Profile update error:', error);
      return { success: false, message: 'Network error, please try again' };
    }
  };

  const changePassword = async (currentPassword: string, newPassword: string) => {
    try {
      if (!user) {
        return { success: false, message: 'You must be logged in' };
      }

      const response = await fetchWithAuth('/user/change-password', {
        method: 'PUT',
        body: JSON.stringify({
          email: user.email,
          currentPassword,
          newPassword,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        return { success: false, message: data.message || 'Password change failed' };
      }

      return { success: true, message: 'Password changed successfully' };
    } catch (error) {
      console.error('Password change error:', error);
      return { success: false, message: 'Network error, please try again' };
    }
  };

  const deleteAccount = async () => {
    try {
      if (!user) {
        return { success: false, message: 'You must be logged in' };
      }

      const response = await fetchWithAuth('/user/delete-account', {
        method: 'DELETE',
        body: JSON.stringify({ email: user.email }),
      });

      if (!response.ok) {
        const data = await response.json();
        return { success: false, message: data.message || 'Account deletion failed' };
      }

      // Clear user data and log out
      await logout();
      return { success: true, message: 'Account deleted successfully' };
    } catch (error) {
      console.error('Account deletion error:', error);
      return { success: false, message: 'Network error, please try again' };
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        login,
        register,
        logout,
        updateProfile,
        changePassword,
        deleteAccount,
        fetchWithAuth
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

// Hook to use the auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Add default export for the AuthProvider
export default AuthProvider;