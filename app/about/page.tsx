import Image from "next/image"
import Link from "next/link"
import { Mail, MapPin, Phone, ChevronRight, Target, Users, Lightbulb, Calendar } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

export default function AboutPage() {
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
            <Link
              href="/about"
              className="text-sm font-medium text-navy-600 border-b-2 border-navy-600 transition-colors"
            >
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
            <Link href="/notice" className="text-sm font-medium text-gray-700 hover:text-navy-600 transition-colors">
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
              src="/placeholder.svg?height=800&width=1600"
              alt="JP SCIENCE Office"
              fill
              className="object-cover brightness-50"
              priority
            />
          </div>
          <div className="container relative z-10 py-24 md:py-32">
            <div className="max-w-3xl space-y-5 text-white">
              <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">JP SCIENCE 소개</h1>
              <p className="text-lg md:text-xl">
                JP SCIENCE는 첨단 과학 기술을 통해 미래를 선도하는 기업입니다. 우리의 혁신적인 연구와 개발은 산업의
                새로운 표준을 만들어갑니다.
              </p>
            </div>
          </div>
        </section>

        {/* CEO Greeting Section */}
        <section id="ceo-greeting" className="py-16 bg-gray-50">
          <div className="container">
            <div className="grid md:grid-cols-2 gap-8 items-center">
              <div className="relative h-[400px] rounded-lg overflow-hidden">
                <Image src="/images/ceo.png" alt="홍진표 대표" fill className="object-cover object-top" />
              </div>
              <div className="space-y-4">
                <h2 className="text-3xl font-bold text-navy-700">대표 인사말</h2>
                <div className="w-20 h-1 bg-navy-600"></div>
                <p className="text-gray-700">안녕하십니까, JP SCIENCE 대표 홍진표입니다.</p>
                <p className="text-gray-700">
                  저희 JP SCIENCE는 2024년 설립 이래로 첨단 과학 기술 분야에서 끊임없는 연구와 혁신을 추구해 왔습니다.
                  우리의 목표는 고객에게 최고 품질의 서비스와 제품을 제공하여 산업 발전에 기여하는 것입니다.
                </p>
                <p className="text-gray-700">
                  전문성과 신뢰를 바탕으로 고객의 요구에 맞는 최적의 솔루션을 제공하기 위해 항상 노력하고 있습니다.
                  앞으로도 JP SCIENCE는 지속적인 연구개발을 통해 혁신적인 기술을 선보이며 글로벌 시장에서 경쟁력을 갖춘
                  기업으로 성장해 나갈 것입니다.
                </p>
                <p className="text-gray-700">
                  저희 JP SCIENCE는 고객과 함께 성장하며, 과학 기술의 발전을 통해 더 나은 미래를 만들어 가고자 합니다.
                  앞으로도 JP SCIENCE에 많은 관심과 성원 부탁드립니다.
                </p>
                <p className="font-semibold text-navy-700">홍진표 드림</p>
              </div>
            </div>
          </div>
        </section>

        {/* Vision Section */}
        <section id="vision" className="py-16">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">비전</h2>
              <p className="mt-4 text-gray-600">JP SCIENCE가 추구하는 가치와 목표</p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              {/* Vision 1 */}
              <Card className="p-6 hover:shadow-lg transition-shadow text-center">
                <div className="w-16 h-16 rounded-full bg-navy-100 flex items-center justify-center mx-auto mb-4">
                  <Target className="h-8 w-8 text-navy-600" />
                </div>
                <h3 className="text-xl font-bold text-navy-700 mb-3">미션</h3>
                <p className="text-gray-700">
                  첨단 과학 기술을 통해 산업 발전에 기여하고, 고객에게 최고 품질의 서비스와 제품을 제공하여 함께
                  성장합니다.
                </p>
              </Card>

              {/* Vision 2 */}
              <Card className="p-6 hover:shadow-lg transition-shadow text-center">
                <div className="w-16 h-16 rounded-full bg-navy-100 flex items-center justify-center mx-auto mb-4">
                  <Lightbulb className="h-8 w-8 text-navy-600" />
                </div>
                <h3 className="text-xl font-bold text-navy-700 mb-3">비전</h3>
                <p className="text-gray-700">
                  혁신적인 연구와 개발을 통해 과학 기술 분야의 선도적인 기업으로 성장하며, 글로벌 시장에서 인정받는
                  기업이 됩니다.
                </p>
              </Card>

              {/* Vision 3 */}
              <Card className="p-6 hover:shadow-lg transition-shadow text-center">
                <div className="w-16 h-16 rounded-full bg-navy-100 flex items-center justify-center mx-auto mb-4">
                  <Users className="h-8 w-8 text-navy-600" />
                </div>
                <h3 className="text-xl font-bold text-navy-700 mb-3">핵심 가치</h3>
                <p className="text-gray-700">
                  전문성, 혁신, 신뢰, 고객 중심, 지속 가능성을 핵심 가치로 삼아 모든 업무에 임하고 있습니다.
                </p>
              </Card>
            </div>

            <div className="mt-16 bg-navy-50 rounded-lg p-8">
              <div className="grid md:grid-cols-2 gap-8 items-center">
                <div>
                  <h3 className="text-2xl font-bold text-navy-700 mb-4">JP SCIENCE의 목표</h3>
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                      <span className="text-gray-700">지속적인 연구개발을 통한 혁신적인 기술 개발 및 특허 확보</span>
                    </li>
                    <li className="flex items-start">
                      <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                      <span className="text-gray-700">
                        고객 맞춤형 솔루션 제공으로 고객 만족도 향상 및 장기적 파트너십 구축
                      </span>
                    </li>
                    <li className="flex items-start">
                      <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                      <span className="text-gray-700">국내외 연구기관 및 기업과의 협력을 통한 기술 경쟁력 강화</span>
                    </li>
                    <li className="flex items-start">
                      <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                      <span className="text-gray-700">친환경적이고 지속 가능한 기술 개발로 사회적 책임 실현</span>
                    </li>
                    <li className="flex items-start">
                      <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                      <span className="text-gray-700">글로벌 시장 진출을 통한 기업 성장 및 브랜드 가치 향상</span>
                    </li>
                  </ul>
                </div>
                <div className="relative h-[300px] rounded-lg overflow-hidden">
                  <Image
                    src="/placeholder.svg?height=300&width=500"
                    alt="JP SCIENCE Vision"
                    fill
                    className="object-cover"
                  />
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* History Section */}
        <section id="history" className="py-16 bg-gray-50">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">연혁</h2>
              <p className="mt-4 text-gray-600">JP SCIENCE의 성장 과정</p>
            </div>

            <div className="relative max-w-3xl mx-auto">
              {/* Vertical line */}
              <div className="absolute left-1/2 transform -translate-x-1/2 h-full w-0.5 bg-navy-200"></div>

              {/* Timeline items */}
              <div className="space-y-16">
                {/* 2025 */}
                <div className="relative">
                  <div className="flex items-center justify-center">
                    <div className="absolute z-10 w-12 h-12 rounded-full bg-navy-600 flex items-center justify-center left-1/2 transform -translate-x-1/2">
                      <Calendar className="h-6 w-6 text-white" />
                    </div>
                  </div>
                  <div className="ml-auto mr-auto md:ml-auto md:mr-0 md:pr-8 w-full md:w-1/2 mt-10 relative">
                    <Card className="p-6 hover:shadow-lg transition-shadow border-l-4 border-l-navy-600">
                      <div className="absolute -left-3 top-4 md:hidden w-6 h-0.5 bg-navy-200"></div>
                      <h3 className="text-xl font-bold text-navy-700">2025</h3>
                      <div className="mt-4 space-y-3">
                        <div className="flex items-start">
                          <span className="text-navy-600 font-semibold mr-2">4월</span>
                          <p className="text-gray-700">한양대학교 캠퍼스타운 사업 선정</p>
                        </div>
                      </div>
                    </Card>
                  </div>
                </div>

                {/* 2024 */}
                <div className="relative">
                  <div className="flex items-center justify-center">
                    <div className="absolute z-10 w-12 h-12 rounded-full bg-navy-600 flex items-center justify-center left-1/2 transform -translate-x-1/2">
                      <Calendar className="h-6 w-6 text-white" />
                    </div>
                  </div>
                  <div className="ml-auto mr-auto md:mr-auto md:ml-0 md:pl-8 w-full md:w-1/2 mt-10 relative">
                    <Card className="p-6 hover:shadow-lg transition-shadow border-l-4 border-l-navy-600">
                      <div className="absolute -right-3 top-4 md:hidden w-6 h-0.5 bg-navy-200"></div>
                      <h3 className="text-xl font-bold text-navy-700">2024</h3>
                      <div className="mt-4 space-y-3">
                        <div className="flex items-start">
                          <span className="text-navy-600 font-semibold mr-2">7월</span>
                          <p className="text-gray-700">JP SCIENCE 설립</p>
                        </div>
                        <div className="flex items-start">
                          <span className="text-navy-600 font-semibold mr-2">4월</span>
                          <p className="text-gray-700">한양대학교 창업중심대학 예비창업 지원사업 선정</p>
                        </div>
                      </div>
                    </Card>
                  </div>
                </div>
              </div>
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
