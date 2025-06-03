import React from 'react';
import { Eye, Brain, AlertTriangle } from 'lucide-react';

const LandingPage = ({ setAuthMode, setShowAuthModal }) => (
  <div className="w-screen min-h-screen bg-white flex">
    {/* Left Side - Image */}
    <div className="hidden md:flex md:w-1/2 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-600/20 to-purple-600/20 z-10"></div>
      <img
        src="https://images.unsplash.com/photo-1449824913935-59a10b8d2000?auto=format&fit=crop&w=800&q=80"
        alt="Driver monitoring dashboard"
        className="w-full h-full object-cover"
      />
      <div className="absolute bottom-8 left-8 right-8 z-20 text-white">
        <div className="bg-black/20 backdrop-blur-sm rounded-xl p-6 border border-white/20">
          <h3 className="text-2xl font-bold mb-2">Real-time Driver Monitoring</h3>
          <p className="text-white/90">Advanced AI technology that keeps drivers safe through continuous monitoring of fatigue indicators.</p>
        </div>
      </div>
    </div>

    {/* Right Side - Content */}
    <div className="w-full md:w-1/2 flex flex-col">
      <header className="flex justify-between items-center px-8 py-6 border-b border-gray-100">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
            <Eye className="w-6 h-6 text-white" />
          </div>
          <span className="text-xl font-bold text-gray-900">MINIPROJ</span>
        </div>
        <div className="flex items-center space-x-4">
          <button className="text-gray-600 hover:text-gray-900 font-medium">Contact</button>
          <button
            onClick={() => { setAuthMode('login'); setShowAuthModal(true); }}
            className="px-6 py-2 bg-gray-900  text-gray-600 rounded-full hover:bg-gray-800 transition-colors font-medium"
          >
            Login / Signup
          </button>
        </div>
      </header>

      <main className="flex-1 flex flex-col justify-center px-8 py-12">
        <div className="max-w-lg">
          <div className="mb-4">
            <span className="inline-block px-3 py-1 bg-blue-100 text-blue-800 text-sm font-medium rounded-full">
              Drowsiness Detection
            </span>
          </div>
          
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6 leading-tight">
            Stay awake, stay safe:
            <span className="block text-blue-600">Real-time alerts for drowsy drivers.</span>
          </h1>
          
          <p className="text-xl text-gray-600 mb-8 leading-relaxed">
            Detect and alert on drowsiness in real-time with industry-leading AI, trained on millions of operational data points to keep drivers safe on the road.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 mb-12">
            <button
              onClick={() => { setAuthMode('signup'); setShowAuthModal(true); }}
              className="px-8 py-4 bg-blue-600 text-gray-900 text-lg font-semibold rounded-xl hover:bg-blue-700 transition-colors shadow-lg"
            >
              Get Started Free
            </button>
            <button
              onClick={() => { setAuthMode('login'); setShowAuthModal(true); }}
              className="px-8 py-4 bg-white text-gray-900 text-lg font-semibold rounded-xl hover:bg-gray-50 transition-colors border-2 border-gray-200"
            >
              View Demo
            </button>
          </div>

          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                <Eye className="w-4 h-4 text-green-600" />
              </div>
              <span className="text-gray-700">Advanced eye tracking & Yawn Detection</span>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
                <Brain className="w-4 h-4 text-purple-600" />
              </div>
              <span className="text-gray-700">Head pose analysis & movement tracking</span>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center">
                <AlertTriangle className="w-4 h-4 text-orange-600" />
              </div>
              <span className="text-gray-700">Instant alerts & fatigue notifications</span>
            </div>
          </div>
        </div>
      </main>
    </div>
  </div>
);

export default LandingPage;
