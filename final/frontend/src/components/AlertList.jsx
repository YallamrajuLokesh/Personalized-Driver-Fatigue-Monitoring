import React from 'react';
import { AlertTriangle, Shield } from 'lucide-react';

const AlertList = ({ alerts }) => (
  <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
    <h2 className="text-xl font-semibold mb-6 text-gray-900 flex items-center">
      <AlertTriangle className="w-5 h-5 mr-2 text-red-600" />
      Recent Alerts
    </h2>
    {alerts.length > 0 ? (
      <div className="space-y-3 max-h-60 overflow-y-auto">
        {alerts.map((alert, index) => (
          <div
            key={alert._id?.$oid || index}
            className="bg-gray-50 p-4 rounded-lg flex justify-between items-center"
          >
            <div>
              <span className="font-medium text-red-600">{alert.type}:</span>
              <span className="ml-2 text-gray-900">
                {alert.value ? alert.value.toFixed(2) : 'N/A'}
              </span>
              {alert.username && (
                <span className="text-gray-500 ml-2">
                  (Driver: {alert.username})
                </span>
              )}
            </div>
            <span className="text-gray-400 text-sm">{alert.time}</span>
          </div>
        ))}
      </div>
    ) : (
      <div className="text-center py-8">
        <Shield className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-500">
          No alerts recorded yet. System is monitoring safely.
        </p>
      </div>
    )}
  </div>
);

export default AlertList;
