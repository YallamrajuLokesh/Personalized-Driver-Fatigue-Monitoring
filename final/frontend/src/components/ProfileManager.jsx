import React from 'react';
import { Users } from 'lucide-react';

const ProfileManager = ({ 
  profiles, 
  selectedProfile, 
  setSelectedProfile, 
  handleSelectProfile, 
  handleRecalibrateProfile 
}) => (
  <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
    <h2 className="text-xl font-semibold mb-6 text-gray-900 flex items-center">
      <Users className="w-5 h-5 mr-2 text-blue-600" />
      Driver Profile
    </h2>
    <div className="space-y-4">
      <label htmlFor="selectProfile" className="block text-sm font-medium text-gray-700">
        Select Profile:
      </label>
      <select
        id="selectProfile"
        className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        value={selectedProfile}
        onChange={(e) => setSelectedProfile(e.target.value)}
      >
        <option value="">-- Select a Profile --</option>
        {profiles.map((profileName) => (
          <option key={profileName} value={profileName}>{profileName}</option>
        ))}
      </select>
      <div className="space-y-2">
        <button
          onClick={handleSelectProfile}
          className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-black font-medium rounded-lg transition-colors"
        >
          Select Profile
        </button>
        <button
          onClick={handleRecalibrateProfile}
          className="w-full py-3 bg-orange-600 hover:bg-orange-700 text-black font-medium rounded-lg transition-colors"
        >
          Recalibrate
        </button>
      </div>
    </div>
  </div>
);

export default ProfileManager;
