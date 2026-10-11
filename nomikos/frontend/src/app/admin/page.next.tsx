"use client";

import { ProtectedRoute } from "../../components/ProtectedRoute";
import { AdminDashboardPage } from "../../pages/AdminDashboardPage";

export default function AdminRoute() {
  return (
    <ProtectedRoute>
      <AdminDashboardPage />
    </ProtectedRoute>
  );
}
