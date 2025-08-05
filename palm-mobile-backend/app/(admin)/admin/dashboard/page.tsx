"use client"

import { useState, useEffect } from "react"
import { toast } from "sonner"
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table"
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogDescription 
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"

type User = {
  _id: string
  name: string
  email: string
  status: "active" | "suspended"
  isAdmin: boolean
  createdAt: string
}

export default function AdminDashboard() {
  const [users, setUsers] = useState<User[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedUser, setSelectedUser] = useState<User | null>(null)
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState("")
  const [statusFilter, setStatusFilter] = useState<string>("all")
  
  // Stats
  const [stats, setStats] = useState({
    total: 0,
    active: 0,
    suspended: 0,
    admins: 0
  })

  useEffect(() => {
    fetchUsers()
  }, [])

  const fetchUsers = async () => {
    try {
      setLoading(true)
      const response = await fetch("/api/admin/users")
      
      if (!response.ok) {
        throw new Error("Failed to fetch users")
      }
      
      const data = await response.json()
      setUsers(data)
      
      // Calculate stats
      const activeUsers = data.filter((user: User) => user.status === "active").length
      const suspendedUsers = data.filter((user: User) => user.status === "suspended").length
      const adminUsers = data.filter((user: User) => user.isAdmin).length
      
      setStats({
        total: data.length,
        active: activeUsers,
        suspended: suspendedUsers,
        admins: adminUsers
      })
    } catch (error) {
      console.error(error)
      toast.error("Failed to load users")
    } finally {
      setLoading(false)
    }
  }

  const handleStatusChange = async (userId: string, newStatus: "active" | "suspended") => {
    try {
      const response = await fetch(`/api/admin/users/${userId}/status`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ status: newStatus }),
      })

      if (!response.ok) {
        throw new Error("Failed to update user status")
      }

      // Update user in the local state
      setUsers(prev => prev.map(user => 
        user._id === userId ? { ...user, status: newStatus } : user
      ))
      
      if (selectedUser && selectedUser._id === userId) {
        setSelectedUser({ ...selectedUser, status: newStatus })
      }
      
      toast.success(`User status updated to ${newStatus}`)
      fetchUsers() // Refresh user list and stats
    } catch (error) {
      console.error(error)
      toast.error("Failed to update user status")
    }
  }

  const openUserDetails = (user: User) => {
    setSelectedUser(user)
    setIsDialogOpen(true)
  }

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          user.email.toLowerCase().includes(searchTerm.toLowerCase())
    
    if (statusFilter === "all") return matchesSearch
    if (statusFilter === "admin") return matchesSearch && user.isAdmin
    return matchesSearch && user.status === statusFilter
  })

  return (
    <div className="container mx-auto p-8 space-y-6">
      <h1 className="text-3xl font-bold">User Management</h1>
      
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Users</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.total}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Active Users</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{stats.active}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Suspended Users</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">{stats.suspended}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Admin Users</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">{stats.admins}</div>
          </CardContent>
        </Card>
      </div>
      
      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1">
          <Input
            placeholder="Search by name or email..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="w-full sm:w-48">
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger>
              <SelectValue placeholder="Filter by status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Users</SelectItem>
              <SelectItem value="active">Active</SelectItem>
              <SelectItem value="suspended">Suspended</SelectItem>
              <SelectItem value="admin">Admins</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <Button onClick={fetchUsers} variant="outline">Refresh</Button>
      </div>
      
      {/* Users Table */}
      {loading ? (
        <div className="flex justify-center p-8">
          <p>Loading users...</p>
        </div>
      ) : (
        <div className="border rounded-lg">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Email</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Role</TableHead>
                <TableHead>Created At</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredUsers.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8">
                    No users found
                  </TableCell>
                </TableRow>
              ) : (
                filteredUsers.map((user) => (
                  <TableRow key={user._id}>
                    <TableCell className="font-medium">{user.name}</TableCell>
                    <TableCell>{user.email}</TableCell>
                    <TableCell>
                      <Badge variant={user.status === "active" ? "success" : "destructive"}>
                        {user.status}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      {user.isAdmin ? (
                        <Badge variant="outline" className="bg-blue-100 text-blue-800 border-blue-300">
                          Admin
                        </Badge>
                      ) : (
                        <Badge variant="outline">User</Badge>
                      )}
                    </TableCell>
                    <TableCell>{new Date(user.createdAt).toLocaleDateString()}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Button 
                          variant="outline" 
                          size="sm" 
                          onClick={() => openUserDetails(user)}
                        >
                          View
                        </Button>
                        {user.status === "active" ? (
                          <Button 
                            variant="destructive" 
                            size="sm" 
                            onClick={() => handleStatusChange(user._id, "suspended")}
                          >
                            Suspend
                          </Button>
                        ) : (
                          <Button 
                            variant="outline" 
                            size="sm" 
                            className="border-green-500 text-green-600 hover:bg-green-50"
                            onClick={() => handleStatusChange(user._id, "active")}
                          >
                            Activate
                          </Button>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      )}
      
      {/* User Details Dialog */}
      {selectedUser && (
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>User Details</DialogTitle>
              <DialogDescription>
                View and manage user information
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="grid grid-cols-[100px_1fr] items-center gap-2">
                <div className="text-sm font-medium">Name:</div>
                <div>{selectedUser.name}</div>
              </div>
              <div className="grid grid-cols-[100px_1fr] items-center gap-2">
                <div className="text-sm font-medium">Email:</div>
                <div>{selectedUser.email}</div>
              </div>
              <div className="grid grid-cols-[100px_1fr] items-center gap-2">
                <div className="text-sm font-medium">Status:</div>
                <div className="flex items-center">
                  <Badge variant={selectedUser.status === "active" ? "success" : "destructive"}>
                    {selectedUser.status}
                  </Badge>
                  <div className="flex-grow"></div>
                  {selectedUser.status === "active" ? (
                    <Button 
                      variant="destructive" 
                      size="sm" 
                      onClick={() => handleStatusChange(selectedUser._id, "suspended")}
                    >
                      Suspend User
                    </Button>
                  ) : (
                    <Button 
                      variant="outline" 
                      size="sm"
                      className="border-green-500 text-green-600 hover:bg-green-50"
                      onClick={() => handleStatusChange(selectedUser._id, "active")}
                    >
                      Activate User
                    </Button>
                  )}
                </div>
              </div>
              <div className="grid grid-cols-[100px_1fr] items-center gap-2">
                <div className="text-sm font-medium">Role:</div>
                <div>
                  {selectedUser.isAdmin ? (
                    <Badge variant="outline" className="bg-blue-100 text-blue-800 border-blue-300">
                      Admin
                    </Badge>
                  ) : (
                    <Badge variant="outline">User</Badge>
                  )}
                </div>
              </div>
              <div className="grid grid-cols-[100px_1fr] items-center gap-2">
                <div className="text-sm font-medium">Joined:</div>
                <div>{new Date(selectedUser.createdAt).toLocaleString()}</div>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  )
}
