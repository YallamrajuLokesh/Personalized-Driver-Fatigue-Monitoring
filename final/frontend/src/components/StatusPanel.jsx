import React from 'react';
import { Activity, Car } from 'lucide-react';

const StatusPanel = ({ currentStatus }) => (
  <div className="xl:col-span-2 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
    <h2 className="text-xl font-semibold mb-6 text-gray-900 flex items-center">
      <Activity className="w-5 h-5 mr-2 text-green-600" />
      Live Driver Status
    </h2>
    {currentStatus ? (
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-sm text-gray-500 mb-1">Active Driver</p>
          <p className="text-lg font-semibold text-gray-900">{currentStatus.current_user || 'None Selected'}</p>
        </div>
        
        <div className={`p-4 rounded-lg ${currentStatus.eye_alarm ? 'bg-red-50 border border-red-200' : 'bg-gray-50'}`}>
          <p className="text-sm text-gray-500 mb-1">Eye Aspect Ratio (EAR)</p>
          <p className={`text-lg font-semibold ${currentStatus.eye_alarm ? 'text-red-600' : 'text-gray-900'}`}>
            {currentStatus.ear ? currentStatus.ear.toFixed(2) : 'N/A'}
          </p>
          <p className="text-xs text-gray-500">Threshold: {currentStatus.EAR_THRESHOLD ? currentStatus.EAR_THRESHOLD.toFixed(2) : '0.21'}</p>
        </div>

        <div className={`p-4 rounded-lg ${currentStatus.yawn_alarm ? 'bg-yellow-50 border border-yellow-200' : 'bg-gray-50'}`}>
          <p className="text-sm text-gray-500 mb-1">Mouth Aspect Ratio (MAR)</p>
          <p className={`text-lg font-semibold ${currentStatus.yawn_alarm ? 'text-yellow-600' : 'text-gray-900'}`}>
            {currentStatus.mar ? currentStatus.mar.toFixed(2) : 'N/A'}
          </p>
          <p className="text-xs text-gray-500">Threshold: {currentStatus.MAR_THRESHOLD ? currentStatus.MAR_THRESHOLD.toFixed(2) : '0.75'}</p>
        </div>

        <div className={`p-4 rounded-lg ${currentStatus.pitch_alarm ? 'bg-red-50 border border-red-200' : 'bg-gray-50'}`}>
          <p className="text-sm text-gray-500 mb-1">Head Pitch</p>
          <p className={`text-lg font-semibold ${currentStatus.pitch_alarm ? 'text-red-600' : 'text-gray-900'}`}>
            {currentStatus.pitch ? currentStatus.pitch.toFixed(2) : 'N/A'}°
          </p>
          <p className="text-xs text-gray-500">Neutral: {currentStatus.NEUTRAL_PITCH ? currentStatus.NEUTRAL_PITCH.toFixed(2) : 'N/A'}°</p>
        </div>

        <div className={`p-4 rounded-lg ${currentStatus.yaw_alarm ? 'bg-red-50 border border-red-200' : 'bg-gray-50'}`}>
          <p className="text-sm text-gray-500 mb-1">Head Yaw</p>
          <p className={`text-lg font-semibold ${currentStatus.yaw_alarm ? 'text-red-600' : 'text-gray-900'}`}>
            {currentStatus.yaw ? currentStatus.yaw.toFixed(2) : 'N/A'}°
          </p>
        </div>

        <div className={`p-4 rounded-lg ${currentStatus.head_tilt_alarm ? 'bg-red-50 border border-red-200' : 'bg-gray-50'}`}>
          <p className="text-sm text-gray-500 mb-1">Head Roll</p>
          <p className={`text-lg font-semibold ${currentStatus.head_tilt_alarm ? 'text-red-600' : 'text-gray-900'}`}>
            {currentStatus.roll ? currentStatus.roll.toFixed(2) : 'N/A'}°
          </p>
          <p className="text-xs text-gray-500">Neutral: {currentStatus.NEUTRAL_ROLL ? currentStatus.NEUTRAL_ROLL.toFixed(2) : 'N/A'}°</p>
        </div>
      </div>
    ) : (
      <div className="text-center py-12">
        <Car className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-500">Waiting for data from monitoring system...</p>
      </div>
    )}
  </div>
);

export default StatusPanel;
