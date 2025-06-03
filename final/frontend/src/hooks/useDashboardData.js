import { useState, useEffect } from 'react';
import { fetchStatus, fetchAlerts, fetchProfiles } from '../api/api';

const useDashboardData = (isAuthenticated, currentUser) => {
  const [currentStatus, setCurrentStatus] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [profiles, setProfiles] = useState([]);
  const [error, setError] = useState(null);
  const [selectedProfile, setSelectedProfile] = useState('');

  useEffect(() => {
    if (!isAuthenticated) return;

    const fetchData = async () => {
      try {
        const [statusData, alertsData, profilesData] = await Promise.all([
          fetchStatus(),
          currentUser?.email ? fetchAlerts(currentUser.email) : [],
          currentUser?.email ? fetchProfiles(currentUser.email) : [],
        ]);

        setCurrentStatus(statusData);
        setAlerts(alertsData);
        setProfiles(profilesData);
        setError(null);
      } catch (e) {
        console.error("Failed to fetch dashboard data:", e);
        setError("Failed to connect to backend API.");
      }
    };

    fetchData();
    const intervals = [
      setInterval(() => fetchStatus().then(setCurrentStatus), 1000),
      setInterval(() => currentUser?.email && fetchAlerts(currentUser.email).then(setAlerts), 2000),
      setInterval(() => currentUser?.email && fetchProfiles(currentUser.email).then(setProfiles), 5000),
    ];

    return () => intervals.forEach(clearInterval);
  }, [isAuthenticated, currentUser]);

  return {
    currentStatus,
    alerts,
    profiles,
    error,
    selectedProfile,
    setSelectedProfile,
  };
};

export default useDashboardData;
