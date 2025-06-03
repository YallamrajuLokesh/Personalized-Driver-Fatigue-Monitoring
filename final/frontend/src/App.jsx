import React, { useState } from 'react';
import AuthModal from './components/AuthModal';
import Dashboard from './components/Dashboard';
import LandingPage from './components/LandingPage';
import useDashboardData from './hooks/useDashboardData';
import { loginUser, registerUser, selectProfile, recalibrateProfile } from './api/api';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState('login');
  const [authForm, setAuthForm] = useState({ name: '', email: '', password: '', confirmPassword: '' });

  const dashboardData = useDashboardData(isAuthenticated, currentUser);

  const handleAuth = async (e) => {
    e.preventDefault();
    try {
      if (authMode === 'signup') {
        if (authForm.password !== authForm.confirmPassword) {
          alert('Passwords do not match!');
          return;
        }
        const data = await registerUser({
          username: authForm.email,
          password: authForm.password,
        });
        setCurrentUser({ name: authForm.name, email: authForm.email });
      } else {
        const data = await loginUser({
          username: authForm.email,
          password: authForm.password,
        });
        setCurrentUser({ email: authForm.email });
      }
      setIsAuthenticated(true);
      setShowAuthModal(false);
      setAuthForm({ name: '', email: '', password: '', confirmPassword: '' });
    } catch (err) {
      alert(`${authMode === 'login' ? 'Login' : 'Signup'} failed. Please try again.`);
    }
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setCurrentUser(null);
    dashboardData.setSelectedProfile('');
  };

  const handleSelectProfile = async () => {
    if (!dashboardData.selectedProfile) {
      alert("Please select a profile from the dropdown.");
      return;
    }
    try {
      await selectProfile(currentUser.email, dashboardData.selectedProfile);
      alert('Profile selected successfully');
    } catch (e) {
      alert(`Error selecting profile: ${e.message}`);
    }
  };

  const handleRecalibrateProfile = async () => {
    if (!dashboardData.currentStatus?.current_user) {
      alert("No user is currently active to recalibrate. Please select one.");
      return;
    }
    try {
      const data = await recalibrateProfile();
      alert(data.message);
    } catch (e) {
      alert(`Error recalibrating: ${e.message}`);
    }
  };

  return (
    <div>
      {!isAuthenticated ? (
        <LandingPage 
          setAuthMode={setAuthMode}
          setShowAuthModal={setShowAuthModal}
        />
      ) : (
        <Dashboard
          currentUser={currentUser}
          handleLogout={handleLogout}
          {...dashboardData}
        />
      )}
      {showAuthModal && (
        <AuthModal
          authMode={authMode}
          setAuthMode={setAuthMode}
          authForm={authForm}
          setAuthForm={setAuthForm}
          handleAuth={handleAuth}
          setShowAuthModal={setShowAuthModal}
        />
      )}
    </div>
  );
}

export default App;