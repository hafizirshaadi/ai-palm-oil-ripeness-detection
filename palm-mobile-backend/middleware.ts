import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'
import { getToken } from 'next-auth/jwt'

export async function middleware(request: NextRequest) {
  const token = await getToken({ req: request })
  const isAdminRoute = request.nextUrl.pathname.startsWith('/admin')

  // If user is not authenticated and tries to access protected routes
  if (!token) {
    if (isAdminRoute || request.nextUrl.pathname.startsWith('/dashboard')) {
      return NextResponse.redirect(new URL('/login', request.url))
    }
    return NextResponse.next()
  }

  // If user is authenticated but suspended
  if (token.status === 'suspended') {
    return NextResponse.redirect(
      new URL('/account-suspended', request.url)
    )
  }

  // If user is trying to access admin routes but isn't an admin
  if (isAdminRoute && !token.isAdmin) {
    return NextResponse.redirect(
      new URL('/unauthorized', request.url)
    )
  }

  // Redirect from login/register pages when already authenticated
  if (
    token && 
    (request.nextUrl.pathname === '/login' || request.nextUrl.pathname === '/register')
  ) {
    const redirectUrl = token.isAdmin ? '/admin/dashboard' : '/dashboard'
    return NextResponse.redirect(new URL(redirectUrl, request.url))
  }

  return NextResponse.next()
}

export const config = {
  matcher: [
    '/dashboard/:path*',
    '/admin/:path*',
    '/login',
    '/register',
  ],
}
