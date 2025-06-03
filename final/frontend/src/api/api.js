const API_BASE_URL = 'http://127.0.0.1:5000';

export const loginUser = async (credentials) => {
  try {
    const response = await fetch(`${API_BASE_URL}/account/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials),
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Login failed');
    }
    
    return data;
  } catch (error) {
    console.error('Login error:', error);
    throw error;
  }
};

export const registerUser = async (userData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/account/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData),
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Registration failed');
    }
    
    return data;
  } catch (error) {
    console.error('Registration error:', error);
    throw error;
  }
};

export const fetchStatus = async () => {
  const response = await fetch(`${API_BASE_URL}/status`);
  if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
  return response.json();
};

export const fetchAlerts = async (email) => {
  const response = await fetch(`${API_BASE_URL}/alerts?account=${encodeURIComponent(email)}`);
  if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
  return response.json();
};

export const fetchProfiles = async (email) => {
  const response = await fetch(`${API_BASE_URL}/profiles?account=${encodeURIComponent(email)}`);
  if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
  return response.json();
};

export const selectProfile = async (accountUsername, profileName) => {
  const response = await fetch(`${API_BASE_URL}/profiles/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ account_username: accountUsername, profile_name: profileName }),
  });
  return response.json();
};

export const recalibrateProfile = async () => {
  const response = await fetch(`${API_BASE_URL}/profiles/recalibrate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  return response.json();
};
