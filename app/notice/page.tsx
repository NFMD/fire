import Image from "next/image"
import Link from "next/link"
import { Mail, MapPin, Phone, Search, Calendar, User, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"

// 더미 공지사항 데이터
const noticeData = [
  {
    id: 1,
    title: "JP SCIENCE 홈페이지 리뉴얼 안내",
    date: "2024-07-15",
    author: "관리자",
    content:
      "안녕하세요, JP SCIENCE입니다. 저희 홈페이지가 새롭게 리뉴얼되었습니다. 더욱 편리한 서비스를 제공하기 위해 최선을 다하겠습니다.",
    views: 245,
  },
  {
    id: 2,
    title: "2024년 하반기 연구개발 프로젝트 공고",
    date: "2024-07-10",
    author: "연구개발팀",
    content:
      "JP SCIENCE에서 2024년 하반기 연구개발 프로젝트를 공고합니다. 관심 있는 기업 및 연구기관의 많은 참여 바랍니다.",
    views: 189,
  },
  {
    id: 3,
    title: "신규 장비 도입 안내",
    date: "2024-06-28",
    author: "장비운영팀",
    content: "JP SCIENCE에서 최신 HR-XRD 장비를 도입하였습니다. 이를 통해 더욱 정밀한 분석 서비스를 제공할 예정입니다.",
    views: 156,
  },
  {
    id: 4,
    title: "하계 휴무 안내",
    date: "2024-06-20",
    author: "관리자",
    content:
      "JP SCIENCE 하계 휴무를 안내드립니다. 휴무 기간: 2024년 8월 1일 ~ 8월 5일. 휴무 기간 동안 문의사항은 이메일로 부탁드립니다.",
    views: 132,
  },
  {
    id: 5,
    title: "Insulation Bonding Wire 신제품 출시",
    date: "2024-06-15",
    author: "제품개발팀",
    content:
      "JP SCIENCE에서 개발한 신제품 Insulation Bonding Wire가 출시되었습니다. 기존 제품보다 내열성과 절연 특성이 향상되었습니다.",
    views: 201,
  },
  {
    id: 6,
    title: "2024년 상반기 연구 성과 보고",
    date: "2024-06-05",
    author: "연구개발팀",
    content:
      "JP SCIENCE 2024년 상반기 연구 성과를 보고합니다. 총 3건의 특허 출원과 5편의 논문 게재 성과를 달성하였습니다.",
    views: 178,
  },
  {
    id: 7,
    title: "산학협력 파트너십 체결 안내",
    date: "2024-05-20",
    author: "대외협력팀",
    content:
      "JP SCIENCE에서 한양대학교 공과대학과 산학협력 파트너십을 체결하였습니다. 이를 통해 공동 연구 및 인재 양성을 추진할 예정입니다.",
    views: 145,
  },
  {
    id: 8,
    title: "채용 공고: 연구원 모집",
    date: "2024-05-10",
    author: "인사팀",
    content: "JP SCIENCE에서 신소재 개발 분야 연구원을 모집합니다. 자세한 내용은 공지사항을 참고해주세요.",
    views: 223,
  },
  {
    id: 9,
    title: "고객 만족도 조사 실시 안내",
    date: "2024-04-25",
    author: "고객지원팀",
    content: "JP SCIENCE 서비스 품질 향상을 위한 고객 만족도 조사를 실시합니다. 많은 참여 부탁드립니다.",
    views: 98,
  },
  {
    id: 10,
    title: "2024년 기술 세미나 개최 안내",
    date: "2024-04-15",
    author: "교육팀",
    content:
      "JP SCIENCE에서 '첨단 소재 기술의 현재와 미래'를 주제로 기술 세미나를 개최합니다. 일시: 2024년 5월 15일, 장소: 한양대학교 HIT 빌딩",
    views: 167,
  },
]

export default function NoticePage() {
  return (
    <div className="flex min-h-screen flex-col">
      {/* Header Section */}
      <header className="sticky top-0 z-50 w-full border-b bg-white">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center gap-2">
            <Link href="/" className="flex items-center">
              <Image src="/images/logo.png" alt="JP SCIENCE Logo" width={120} height={40} className="h-10 w-auto" />
              <span className="ml-2 text-xl font-bold text-navy-700">JP SCIENCE</span>
            </Link>
          </div>
          <nav className="hidden md:flex gap-6">
            <Link href="/about" className="text-sm font-medium text-gray-700 hover:text-navy-600 transition-colors">
              회사소개
            </Link>
            <Link href="/research" className="text-sm font-medium text-gray-700 hover:text-navy-600 transition-colors">
              연구개발
            </Link>
            <Link href="/services" className="text-sm font-medium text-gray-700 hover:text-navy-600 transition-colors">
              서비스
            </Link>
            <Link href="/products" className="text-sm font-medium text-gray-700 hover:text-navy-600 transition-colors">
              제품
            </Link>
            <Link
              href="/notice"
              className="text-sm font-medium text-navy-600 border-b-2 border-navy-600 transition-colors"
            >
              공지사항
            </Link>
            <Link href="/contact" className="text-sm font-medium text-gray-700 hover:text-navy-600 transition-colors">
              문의하기
            </Link>
          </nav>
          <Button variant="outline" size="sm" className="md:hidden">
            <span className="sr-only">Toggle menu</span>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-4 w-4"
            >
              <line x1="4" x2="20" y1="12" y2="12" />
              <line x1="4" x2="20" y1="6" y2="6" />
              <line x1="4" x2="20" y1="18" y2="18" />
            </svg>
          </Button>
        </div>
      </header>

      <main className="flex-1">
        {/* Hero Section */}
        <section className="relative">
          <div className="absolute inset-0 z-0">
            <Image
              src="/placeholder.svg?height=400&width=1600"
              alt="Notice Board"
              fill
              className="object-cover brightness-50"
              priority
            />
          </div>
          <div className="container relative z-10 py-16 md:py-24">
            <div className="max-w-3xl space-y-5 text-white">
              <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">공지사항</h1>
              <p className="text-lg md:text-xl">JP SCIENCE의 최신 소식과 중요 공지사항을 확인하세요.</p>
            </div>
          </div>
        </section>

        {/* Notice Board Section */}
        <section className="py-16">
          <div className="container">
            <div className="mb-8 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div className="flex items-center space-x-2">
                <span className="text-gray-600">총 {noticeData.length}건</span>
                <span className="text-gray-400">|</span>
                <span className="text-gray-600">1 페이지</span>
              </div>
              <div className="flex w-full md:w-auto">
                <div className="relative flex-grow md:w-64">
                  <Input placeholder="검색어를 입력하세요" className="pr-10" />
                  <Button
                    size="icon"
                    variant="ghost"
                    className="absolute right-0 top-0 h-full px-3 text-gray-500 hover:text-navy-600"
                  >
                    <Search className="h-4 w-4" />
                    <span className="sr-only">Search</span>
                  </Button>
                </div>
              </div>
            </div>

            <Card className="overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-navy-50 text-left">
                      <th className="whitespace-nowrap px-4 py-3 text-sm font-medium text-navy-700">번호</th>
                      <th className="whitespace-nowrap px-4 py-3 text-sm font-medium text-navy-700">제목</th>
                      <th className="whitespace-nowrap px-4 py-3 text-sm font-medium text-navy-700">작성자</th>
                      <th className="whitespace-nowrap px-4 py-3 text-sm font-medium text-navy-700">작성일</th>
                      <th className="whitespace-nowrap px-4 py-3 text-sm font-medium text-navy-700">조회수</th>
                    </tr>
                  </thead>
                  <tbody>
                    {noticeData.map((notice) => (
                      <tr key={notice.id} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="whitespace-nowrap px-4 py-4 text-sm text-gray-700">{notice.id}</td>
                        <td className="px-4 py-4">
                          <Link
                            href={`/notice/${notice.id}`}
                            className="text-navy-700 hover:text-navy-500 hover:underline font-medium"
                          >
                            {notice.title}
                          </Link>
                        </td>
                        <td className="whitespace-nowrap px-4 py-4 text-sm text-gray-700">
                          <div className="flex items-center">
                            <User className="mr-2 h-4 w-4 text-gray-500" />
                            {notice.author}
                          </div>
                        </td>
                        <td className="whitespace-nowrap px-4 py-4 text-sm text-gray-700">
                          <div className="flex items-center">
                            <Calendar className="mr-2 h-4 w-4 text-gray-500" />
                            {notice.date}
                          </div>
                        </td>
                        <td className="whitespace-nowrap px-4 py-4 text-sm text-gray-700">{notice.views}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            {/* Pagination */}
            <div className="mt-8 flex justify-center">
              <nav className="flex items-center space-x-1">
                <Button variant="outline" size="icon" className="h-8 w-8">
                  <ChevronRight className="h-4 w-4 rotate-180" />
                  <span className="sr-only">Previous Page</span>
                </Button>
                <Button variant="outline" size="sm" className="h-8 w-8 bg-navy-600 text-white hover:bg-navy-700">
                  1
                </Button>
                <Button variant="outline" size="sm" className="h-8 w-8">
                  2
                </Button>
                <Button variant="outline" size="sm" className="h-8 w-8">
                  3
                </Button>
                <Button variant="outline" size="sm" className="h-8 w-8">
                  4
                </Button>
                <Button variant="outline" size="sm" className="h-8 w-8">
                  5
                </Button>
                <Button variant="outline" size="icon" className="h-8 w-8">
                  <ChevronRight className="h-4 w-4" />
                  <span className="sr-only">Next Page</span>
                </Button>
              </nav>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-navy-800 text-white py-12">
        <div className="container">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="mb-4 flex items-center">
                <Image
                  src="/images/logo.png"
                  alt="JP SCIENCE Logo"
                  width={120}
                  height={40}
                  className="h-10 w-auto invert"
                />
                <span className="ml-2 text-xl font-bold text-white">JP SCIENCE</span>
              </div>
              <p className="text-gray-300">끊임없는 선구적인 연구개발과 혁신을 통해 미래를 선도합니다.</p>
            </div>
            <div>
              <h3 className="text-lg font-bold mb-4">바로가기</h3>
              <ul className="space-y-2">
                <li>
                  <Link href="/about" className="text-gray-300 hover:text-white transition-colors">
                    회사소개
                  </Link>
                </li>
                <li>
                  <Link href="/research" className="text-gray-300 hover:text-white transition-colors">
                    연구개발
                  </Link>
                </li>
                <li>
                  <Link href="/services" className="text-gray-300 hover:text-white transition-colors">
                    서비스
                  </Link>
                </li>
                <li>
                  <Link href="/products" className="text-gray-300 hover:text-white transition-colors">
                    제품
                  </Link>
                </li>
                <li>
                  <Link href="/notice" className="text-gray-300 hover:text-white transition-colors">
                    공지사항
                  </Link>
                </li>
                <li>
                  <Link href="/contact" className="text-gray-300 hover:text-white transition-colors">
                    문의하기
                  </Link>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-bold mb-4">연락처</h3>
              <ul className="space-y-2">
                <li className="flex items-center">
                  <MapPin className="h-4 w-4 mr-2" />
                  <span className="text-gray-300">서울특별시 성동구 왕십리로 222 한양대학교 자연과학관 428호</span>
                </li>
                <li className="flex items-center">
                  <Phone className="h-4 w-4 mr-2" />
                  <span className="text-gray-300">02-2220-0911</span>
                </li>
                <li className="flex items-center">
                  <Mail className="h-4 w-4 mr-2" />
                  <span className="text-gray-300">info@jpscience.com</span>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-bold mb-4">뉴스레터 구독</h3>
              <p className="text-gray-300 mb-4">JP SCIENCE의 최신 소식을 받아보세요.</p>
              <div className="flex">
                <input
                  type="email"
                  placeholder="이메일을 입력하세요"
                  className="bg-navy-700 border-navy-600 text-white px-3 py-2 rounded-l-md w-full"
                />
                <Button className="ml-0 bg-navy-600 hover:bg-navy-500 rounded-l-none">구독</Button>
              </div>
            </div>
          </div>
          <div className="border-t border-navy-700 mt-8 pt-8 text-center">
            <p className="text-gray-400">&copy; {new Date().getFullYear()} JP SCIENCE. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
