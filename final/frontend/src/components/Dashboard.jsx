import React from 'react';
import { Eye, UserCheck, LogOut } from 'lucide-react';
import ProfileManager from './ProfileManager';
import StatusPanel from './StatusPanel';
import AlertList from './AlertList';

const Dashboard = ({ currentUser, handleLogout, error, ...props }) => {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <header className="bg-white shadow-sm border-b border-gray-200 w-full">
        <div className="w-full max-w-none px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <Eye className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-2xl font-bold text-gray-900">MINIPROJ Dashboard</h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-gray-600">
                <UserCheck className="w-4 h-4" />
                <span>{currentUser?.name || currentUser?.email}</span>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center space-x-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
              >
                <LogOut className="w-4 h-4" />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="w-full px-6 py-8 max-w-none">
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-800 p-4 rounded-lg mb-6">
            {error} Please ensure Flask API (`api.py`) and MongoDB are running.
          </div>
        )}

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 mb-8">
          <ProfileManager {...props} />
          <StatusPanel currentStatus={props.currentStatus} />
        </div>

        <AlertList alerts={props.alerts} />
      </div>
    </div>
  );
};

export default Dashboard;
