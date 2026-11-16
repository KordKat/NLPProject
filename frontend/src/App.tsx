import { useState } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Menu } from "lucide-react";
import { Logo, Spinner } from "./components/ui";
import AuthPage from "./features/auth/AuthPage";
import Landing from "./features/projects/Landing";
import NewProjectDialog from "./features/projects/NewProjectDialog";
import ProjectWorkspace from "./features/projects/ProjectWorkspace";
import Sidebar from "./features/projects/Sidebar";
import SettingsDialog from "./features/settings/SettingsDialog";
import { AuthProvider, useAuth } from "./hooks/useAuth";
import { ProjectsProvider } from "./hooks/useProjects";

function Shell() {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [newOpen, setNewOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const sidebarProps = {
    onNewProject: () => {
      setNewOpen(true);
      setMobileOpen(false);
    },
    onOpenSettings: () => {
      setSettingsOpen(true);
      setMobileOpen(false);
    },
  };

  return (
    <div className="flex h-full overflow-hidden">
      {/* desktop sidebar */}
      <div className="hidden lg:flex">
        <Sidebar collapsed={collapsed} onToggleCollapsed={() => setCollapsed((c) => !c)} {...sidebarProps} />
      </div>
      {/* mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <Sidebar collapsed={false} onToggleCollapsed={() => setMobileOpen(false)} onNavigate={() => setMobileOpen(false)} {...sidebarProps} />
          <div className="flex-1 bg-black/60" onClick={() => setMobileOpen(false)} aria-hidden />
        </div>
      )}

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-white/5 px-3 py-2 lg:hidden">
          <button className="icon-btn" onClick={() => setMobileOpen(true)} aria-label="Open menu"><Menu size={18} /></button>
          <Logo compact />
          <span className="font-semibold text-slate-100">STEM OS</span>
        </div>
        <div className="min-h-0 flex-1">
          <Routes>
            <Route path="/" element={<Landing onNewProject={() => setNewOpen(true)} />} />
            <Route path="/p/:projectId" element={<ProjectWorkspace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </main>

      <NewProjectDialog open={newOpen} onClose={() => setNewOpen(false)} />
      <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}

function Gate() {
  const { user, loading } = useAuth();
  if (loading) {
    return (
      <div className="grid h-full place-items-center">
        <Spinner className="h-6 w-6" />
      </div>
    );
  }
  if (!user) return <AuthPage />;
  return (
    <ProjectsProvider key={user.id}>
      <Shell />
    </ProjectsProvider>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Gate />
      </AuthProvider>
    </BrowserRouter>
  );
}
