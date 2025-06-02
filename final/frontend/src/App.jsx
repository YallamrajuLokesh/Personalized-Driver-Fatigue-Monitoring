import React, { useState, useEffect } from 'react';
import './index.css'; // Import the main CSS file for Tailwind directives

function App() {
  const [currentStatus, setCurrentStatus] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [profiles, setProfiles] = useState([]);
  const [newUsername, setNewUsername] = useState('');
  const [selectedProfile, setSelectedProfile] = useState('');
  const [error, setError] = useState(null);

  const API_BASE_URL = 'http://127.0.0.1:5000'; // Flask API URL

  // --- Fetch Status, Alerts, and Profiles ---
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/status`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        setCurrentStatus(data);
        // Set selected profile if one is active from the status file
        if (data.current_user && data.current_user !== selectedProfile) {
          setSelectedProfile(data.current_user);
        }
      } catch (e) {
        console.error("Failed to fetch status:", e);
        setError("Failed to connect to backend API for status.");
      }
    };

    const fetchAlerts = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/alerts`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        setAlerts(data);
      } catch (e) {
        console.error("Failed to fetch alerts:", e);
        setError("Failed to connect to backend API for alerts.");
      }
    };

    const fetchProfiles = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/profiles`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        setProfiles(data);
      } catch (e) {
        console.error("Failed to fetch profiles:", e);
        setError("Failed to connect to backend API for profiles.");
      }
    };

    // Initial fetch
    fetchStatus();
    fetchAlerts();
    fetchProfiles();

    // Set up intervals for periodic fetching
    const statusInterval = setInterval(fetchStatus, 1000); // Every 1 second
    const alertsInterval = setInterval(fetchAlerts, 2000); // Every 2 seconds
    const profilesInterval = setInterval(fetchProfiles, 5000); // Every 5 seconds (less frequent)

    // Cleanup intervals on component unmount
    return () => {
      clearInterval(statusInterval);
      clearInterval(alertsInterval);
      clearInterval(profilesInterval);
    };
  }, [selectedProfile]); // Rerun if selectedProfile changes to ensure immediate sync

  // --- Profile Management Functions ---
  const handleCreateProfile = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/profiles/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: newUsername }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Failed to create profile');
      alert(data.message);
      setNewUsername('');
      // Profiles will refetch automatically due to interval
    } catch (e) {
      console.error("Error creating profile:", e.message);
      alert(`Error creating profile: ${e.message}`);
    }
  };

  const handleSelectProfile = async () => {
    if (!selectedProfile) {
      alert("Please select a profile from the dropdown.");
      return;
    }
    try {
      const response = await fetch(`${API_BASE_URL}/profiles/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: selectedProfile }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Failed to select profile');
      alert(data.message);
      // Status will refetch and update current_user
    } catch (e) {
      console.error("Error selecting profile:", e.message);
      alert(`Error selecting profile: ${e.message}`);
    }
  };

  const handleRecalibrateProfile = async () => {
    if (!currentStatus || !currentStatus.current_user) {
      alert("No user is currently active to recalibrate. Please select one.");
      return;
    }
    try {
      const response = await fetch(`${API_BASE_URL}/profiles/recalibrate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // No body needed, main.py will recalibrate the active user
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Failed to send recalibration command');
      alert(data.message);
    } catch (e) {
      console.error("Error sending recalibration command:", e.message);
      alert(`Error recalibrating: ${e.message}`);
    }
  };


  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-6 flex flex-col items-center">
      <h1 className="text-4xl font-bold mb-8 text-blue-400">Driver Fatigue Detection Dashboard</h1>

      {error && (
        <div className="bg-red-800 text-white p-4 rounded-lg mb-6 w-full max-w-4xl text-center">
          {error} Please ensure Flask API (`api.py`) and MongoDB are running.
        </div>
      )}

      {/* User Profile Management */}
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8 w-full max-w-4xl">
        <h2 className="text-2xl font-semibold mb-4 text-purple-300">User Profile Management</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Create Profile */}
          <div>
            <label htmlFor="newUsername" className="block text-gray-300 text-sm font-bold mb-2">
              New Username:
            </label>
            <input
              type="text"
              id="newUsername"
              className="shadow appearance-none border border-gray-700 rounded w-full py-2 px-3 text-gray-900 leading-tight focus:outline-none focus:shadow-outline bg-gray-200"
              value={newUsername}
              onChange={(e) => setNewUsername(e.target.value)}
              placeholder="e.g., JaneDoe"
            />
            <button
              onClick={handleCreateProfile}
              className="mt-4 bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline transition duration-200"
            >
              Create & Select Profile
            </button>
          </div>

          {/* Select Profile */}
          <div>
            <label htmlFor="selectProfile" className="block text-gray-300 text-sm font-bold mb-2">
              Select Existing Profile:
            </label>
            <select
              id="selectProfile"
              className="shadow border border-gray-700 rounded w-full py-2 px-3 text-gray-900 leading-tight focus:outline-none focus:shadow-outline bg-gray-200"
              value={selectedProfile}
              onChange={(e) => setSelectedProfile(e.target.value)}
            >
              <option value="">-- Select a Profile --</option>
              {profiles.map((profileName) => (
                <option key={profileName} value={profileName}>{profileName}</option>
              ))}
            </select>
            <div className="flex gap-4 mt-4">
              <button
                onClick={handleSelectProfile}
                className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline transition duration-200"
              >
                Select Profile
              </button>
              <button
                onClick={handleRecalibrateProfile}
                className="bg-yellow-600 hover:bg-yellow-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline transition duration-200"
              >
                Recalibrate Active Profile
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Live Status Display */}
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8 w-full max-w-4xl">
        <h2 className="text-2xl font-semibold mb-4 text-cyan-300">Live Driver Status</h2>
        {currentStatus ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-lg">
            <p><span className="font-semibold text-gray-300">Active User:</span> {currentStatus.current_user || 'None Selected'}</p>
            <p className={`${currentStatus.eye_alarm ? 'text-red-500 font-bold' : 'text-gray-300'}`}>
              <span className="font-semibold">EAR:</span> {currentStatus.ear ? currentStatus.ear.toFixed(2) : 'N/A'} (Thresh: {currentStatus.EAR_THRESHOLD ? currentStatus.EAR_THRESHOLD.toFixed(2) : '0.21'})
            </p>
            <p className={`${currentStatus.yawn_alarm ? 'text-yellow-500 font-bold' : 'text-gray-300'}`}>
              <span className="font-semibold">MAR:</span> {currentStatus.mar ? currentStatus.mar.toFixed(2) : 'N/A'} (Thresh: {currentStatus.MAR_THRESHOLD ? currentStatus.MAR_THRESHOLD.toFixed(2) : '0.75'})
            </p>
            <p className="text-gray-300"><span className="font-semibold">Mouth Area:</span> {currentStatus.mouth_area ? currentStatus.mouth_area.toFixed(0) : 'N/A'}</p>
            <p className="text-gray-300"><span className="font-semibold">Mouth Ratio:</span> {currentStatus.mouth_ratio ? currentStatus.mouth_ratio.toFixed(2) : 'N/A'}</p>

            <p className={`${currentStatus.pitch_alarm ? 'text-red-500 font-bold' : 'text-gray-300'}`}>
              <span className="font-semibold">Pitch:</span> {currentStatus.pitch ? currentStatus.pitch.toFixed(2) : 'N/A'}° (Neutral: {currentStatus.NEUTRAL_PITCH ? currentStatus.NEUTRAL_PITCH.toFixed(2) : 'N/A'})
            </p>
            <p className={`${currentStatus.yaw_alarm ? 'text-red-500 font-bold' : 'text-gray-300'}`}>
              <span className="font-semibold">Yaw:</span> {currentStatus.yaw ? currentStatus.yaw.toFixed(2) : 'N/A'}°
            </p>
            <p className={`${currentStatus.head_tilt_alarm ? 'text-red-500 font-bold' : 'text-gray-300'}`}>
              <span className="font-semibold">Roll:</span> {currentStatus.roll ? currentStatus.roll.toFixed(2) : 'N/A'}° (Neutral: {currentStatus.NEUTRAL_ROLL ? currentStatus.NEUTRAL_ROLL.toFixed(2) : 'N/A'})
            </p>
          </div>
        ) : (
          <p className="text-gray-400">Waiting for data from `main.py`...</p>
        )}
      </div>

      {/* Alerts Log */}
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-full max-w-4xl">
        <h2 className="text-2xl font-semibold mb-4 text-red-400">Recent Alerts</h2>
        {alerts.length > 0 ? (
          <ul className="space-y-2 max-h-60 overflow-y-auto pr-2">
            {alerts.map((alert, index) => (
              <li key={alert._id.$oid || index} className="bg-gray-700 p-3 rounded-md flex justify-between items-center text-sm">
                <span>
                  <span className="font-semibold text-red-300">{alert.type}:</span> {alert.value ? alert.value.toFixed(2) : 'N/A'}
                  {alert.username && <span className="text-gray-400 ml-2">(User: {alert.username})</span>}
                </span>
                <span className="text-gray-400">{alert.time}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-gray-400">No alerts recorded yet for the active user.</p>
        )}
      </div>
    </div>
  );
}

export default App;