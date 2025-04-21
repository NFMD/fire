import { type NextRequest, NextResponse } from "next/server"
import { jwtVerify } from "jose"

const SECRET_KEY = process.env.JWT_SECRET_KEY || "default_jwt_secret_key_for_development"

export async function middleware(request: NextRequest) {
  const path = request.nextUrl.pathname

  // /admin 경로 접근 시 인증 필요
  if (path.startsWith("/admin") && path !== "/admin/login" && path !== "/admin/setup") {
    const token = request.cookies.get("auth_token")?.value

    // 토큰 없으면 로그인 페이지로 리디렉션
    if (!token) {
      return NextResponse.redirect(new URL("/admin/login", request.url))
    }

    try {
      // JWT 토큰 검증
      const { payload } = await jwtVerify(token, new TextEncoder().encode(SECRET_KEY))

      // 관리자 권한 확인
      if (payload.role !== "ADMIN") {
        return NextResponse.redirect(new URL("/admin/login", request.url))
      }

      return NextResponse.next()
    } catch (error) {
      // 토큰이 잘못되었거나 만료된 경우
      return NextResponse.redirect(new URL("/admin/login", request.url))
    }
  }

  return NextResponse.next()
}

export const config = {
  matcher: ["/admin/:path*"],
}
