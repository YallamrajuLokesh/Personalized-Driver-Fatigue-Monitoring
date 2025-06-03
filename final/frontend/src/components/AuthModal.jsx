import React from 'react';
import { User, Mail, Lock } from 'lucide-react';

const AuthModal = ({
  authMode,
  setAuthMode,
  authForm,
  setAuthForm,
  handleAuth,
  setShowAuthModal,
}) => (
  <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
    <div className="bg-white rounded-2xl p-8 w-full max-w-md shadow-2xl relative">
      <div className="flex justify-center mb-6">
        <div className="flex bg-gray-100 rounded-lg p-1">
          <button
            onClick={() => setAuthMode('login')}
            className={`px-6 py-2 rounded-md font-medium transition-colors ${
              authMode === 'login'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Login
          </button>
          <button
            onClick={() => setAuthMode('signup')}
            className={`px-6 py-2 rounded-md font-medium transition-colors ${
              authMode === 'signup'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Signup
          </button>
        </div>
      </div>

      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          {authMode === 'login' ? 'Welcome back' : 'Create account'}
        </h2>
        <p className="text-gray-600">
          {authMode === 'login' ? 'Sign in to access your dashboard' : 'Join SafeDrive AI today'}
        </p>
      </div>

      <form onSubmit={handleAuth} className="space-y-4">
        {authMode === 'signup' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Full Name</label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                required
                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={authForm.name}
                onChange={(e) => setAuthForm({...authForm, name: e.target.value})}
                placeholder="Enter your name"
              />
            </div>
          </div>
        )}
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Email</label>
          <div className="relative">
            <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="email"
              required
              className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              value={authForm.email}
              onChange={(e) => setAuthForm({...authForm, email: e.target.value})}
              placeholder="Enter your email"
            />
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
          <div className="relative">
            <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="password"
              required
              className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              value={authForm.password}
              onChange={(e) => setAuthForm({...authForm, password: e.target.value})}
              placeholder="Enter your password"
            />
          </div>
        </div>

        {authMode === 'signup' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Confirm Password</label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="password"
                required
                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={authForm.confirmPassword}
                onChange={(e) => setAuthForm({...authForm, confirmPassword: e.target.value})}
                placeholder="Confirm your password"
              />
            </div>
          </div>
        )}

        <button
          type="submit"
          className="w-full py-3 bg-blue-600 text-black font-semibold rounded-lg hover:bg-blue-700 transition-colors"
        >
          {authMode === 'login' ? 'Sign In' : 'Create Account'}
        </button>
      </form>

      <button
        onClick={() => setShowAuthModal(false)}
        className="absolute top-4 right-4 text-black hover:text-gray-600"
      >
        ✕
      </button>
    </div>
  </div>
);

export default AuthModal;
