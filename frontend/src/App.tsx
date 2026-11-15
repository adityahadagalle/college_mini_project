import React, { useState, Suspense, lazy } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { NotificationProvider } from './context/NotificationContext';
import { ToastContainer } from './components/common/ToastContainer';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import { OfflineBanner } from './components/common/OfflineBanner';
import { SkeletonLoader } from './components/common/SkeletonLoader';
import { Navbar } from './components/layout/Navbar';
import { Sidebar } from './components/layout/Sidebar';
import { LandingPage } from './components/dashboard/LandingPage';
import { DashboardView } from './components/dashboard/DashboardView';
import { SinglePredict } from './components/predict/SinglePredict';
import { FileUploader } from './components/batch/FileUploader';
import { HistoryTable } from './components/history/HistoryTable';
import { ModelMetricsView } from './components/dashboard/ModelMetricsView';
import { SystemHealthGauges } from './components/health/SystemHealthGauges';
import { SettingsPanel } from './components/settings/SettingsPanel';
import { AuthModal } from './components/auth/AuthModal';
import { UserProfileModal } from './components/auth/UserProfileModal';

// Lazy load heavy chart-dependent views for bundle optimization
const AnalyticsDashboard = lazy(() =>
  import('./components/analytics/AnalyticsDashboard').then((module) => ({
    default: module.AnalyticsDashboard,
  }))
);

const MlOpsDashboard = lazy(() =>
  import('./components/mlops/MlOpsDashboard').then((module) => ({
    default: module.MlOpsDashboard,
  }))
);

const AspectExplorer = lazy(() =>
  import('./components/absa/AspectExplorer').then((module) => ({
    default: module.AspectExplorer,
  }))
);

const EmotionAnalytics = lazy(() =>
  import('./components/emotion/EmotionAnalytics').then((module) => ({
    default: module.EmotionAnalytics,
  }))
);

const TopicExplorer = lazy(() =>
  import('./components/topics/TopicExplorer').then((module) => ({
    default: module.TopicExplorer,
  }))
);

const ReviewIntelligence = lazy(() =>
  import('./components/intelligence/ReviewIntelligence').then((module) => ({
    default: module.ReviewIntelligence,
  }))
);

const CrossFeatureAnalytics = lazy(() =>
  import('./components/intelligence/CrossFeatureAnalytics').then((module) => ({
    default: module.CrossFeatureAnalytics,
  }))
);

const MainLayout: React.FC = () => {
  const { isAuthenticated, loading } = useAuth();
  const [currentTab, setCurrentTab] = useState<string>('intelligence');
  const [profileModalOpen, setProfileModalOpen] = useState<boolean>(false);
  const [authModalOpen, setAuthModalOpen] = useState<boolean>(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);

  if (loading) {
    return (
      <div className="min-h-screen bg-transparent flex items-center justify-center text-zinc-400 text-xs font-mono animate-pulse">
        Initializing SentimentAI Enterprise Intelligence Engines...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-transparent text-zinc-100 flex flex-col font-sans antialiased selection:bg-zinc-800 selection:text-white">
      <OfflineBanner />
      <ToastContainer />

      <Navbar
        onOpenProfile={() => {
          if (isAuthenticated) {
            setProfileModalOpen(true);
          } else {
            setAuthModalOpen(true);
          }
        }}
        mobileMenuOpen={mobileMenuOpen}
        onToggleMobileMenu={() => setMobileMenuOpen(!mobileMenuOpen)}
      />

      {authModalOpen && <AuthModal onClose={() => setAuthModalOpen(false)} />}
      <UserProfileModal isOpen={profileModalOpen} onClose={() => setProfileModalOpen(false)} />

      <div className="flex flex-1 min-h-[calc(100vh-52px)]">
        <Sidebar
          currentTab={currentTab}
          setCurrentTab={setCurrentTab}
          mobileMenuOpen={mobileMenuOpen}
          onCloseMobileMenu={() => setMobileMenuOpen(false)}
        />

        <main className="flex-1 p-4 sm:p-6 overflow-y-auto w-full">
          <ErrorBoundary>
            <Suspense fallback={<SkeletonLoader variant="chart" count={2} />}>
              {currentTab === 'landing' && <LandingPage onGetStarted={() => setCurrentTab('intelligence')} />}
              {currentTab === 'intelligence' && <ReviewIntelligence onNavigate={(tab) => setCurrentTab(tab)} />}
              {currentTab === 'cross-feature' && <CrossFeatureAnalytics />}
              {currentTab === 'dashboard' && <DashboardView onNavigate={(tab) => setCurrentTab(tab)} />}
              {currentTab === 'analytics' && <AnalyticsDashboard />}
              {currentTab === 'mlops' && <MlOpsDashboard />}
              {currentTab === 'predict' && <SinglePredict />}
              {currentTab === 'absa' && <AspectExplorer />}
              {currentTab === 'emotion' && <EmotionAnalytics />}
              {currentTab === 'topics' && <TopicExplorer />}
              {currentTab === 'batch' && <FileUploader />}
              {currentTab === 'history' && <HistoryTable />}
              {currentTab === 'model' && <ModelMetricsView />}
              {currentTab === 'health' && <SystemHealthGauges />}
              {currentTab === 'settings' && <SettingsPanel />}
            </Suspense>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <AuthProvider>
      <NotificationProvider>
        <MainLayout />
      </NotificationProvider>
    </AuthProvider>
  );
};

export default App;
