import Image from "next/image"
import Link from "next/link"
import { Mail, MapPin, Phone, ChevronRight, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function ProductsPage() {
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
            <Link
              href="/products"
              className="text-sm font-medium text-navy-600 border-b-2 border-navy-600 transition-colors"
            >
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
              alt="Product Showcase"
              fill
              className="object-cover brightness-50"
              priority
            />
          </div>
          <div className="container relative z-10 py-24 md:py-32">
            <div className="max-w-3xl space-y-5 text-white">
              <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">혁신적인 제품</h1>
              <p className="text-lg md:text-xl">
                JP SCIENCE는 첨단 과학 기술을 바탕으로 고품질의 제품을 개발하고 있습니다. 우리의 제품은 고객의 요구에
                맞는 최적의 솔루션을 제공합니다.
              </p>
            </div>
          </div>
        </section>

        {/* Main Product Section */}
        <section className="py-16">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">주요 제품</h2>
              <p className="mt-4 text-gray-600">JP SCIENCE의 대표 제품을 소개합니다</p>
            </div>

            <div className="grid md:grid-cols-2 gap-8 items-center">
              <div className="relative h-[400px] rounded-lg overflow-hidden">
                <Image
                  src="/placeholder.svg?height=400&width=600"
                  alt="Insulation Bonding Wire"
                  fill
                  className="object-cover"
                />
              </div>
              <div className="space-y-4">
                <h3 className="text-2xl font-bold text-navy-700">Insulation Bonding Wire</h3>
                <div className="w-20 h-1 bg-navy-600"></div>
                <p className="text-gray-700">
                  JP SCIENCE의 대표 제품인 Insulation Bonding Wire는 반도체 패키징 공정에서 사용되는 고성능
                  와이어입니다. 우수한 절연 특성과 내구성을 갖추고 있어 고집적 반도체 제품에 최적화되어 있습니다.
                </p>
                <ul className="space-y-2">
                  <li className="flex items-start">
                    <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                    <span>우수한 절연 특성</span>
                  </li>
                  <li className="flex items-start">
                    <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                    <span>높은 내열성 및 내구성</span>
                  </li>
                  <li className="flex items-start">
                    <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                    <span>미세 피치 적용 가능</span>
                  </li>
                  <li className="flex items-start">
                    <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                    <span>다양한 크기 및 사양 맞춤 제작</span>
                  </li>
                </ul>
                <div className="pt-4">
                  <Button className="bg-navy-600 hover:bg-navy-700 text-white">
                    제품 상세 정보 <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Product Specifications Section */}
        <section className="py-16 bg-gray-50">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">제품 사양</h2>
              <p className="mt-4 text-gray-600">Insulation Bonding Wire의 상세 사양</p>
            </div>

            <Tabs defaultValue="specs" className="w-full">
              <TabsList className="grid w-full grid-cols-3 mb-8">
                <TabsTrigger value="specs">기술 사양</TabsTrigger>
                <TabsTrigger value="applications">적용 분야</TabsTrigger>
                <TabsTrigger value="comparison">경쟁 제품 비교</TabsTrigger>
              </TabsList>
              <TabsContent value="specs">
                <Card className="p-6">
                  <div className="grid md:grid-cols-2 gap-8">
                    <div>
                      <h3 className="text-xl font-bold text-navy-700 mb-4">물리적 특성</h3>
                      <ul className="space-y-3">
                        <li className="flex justify-between">
                          <span className="font-medium">직경 범위</span>
                          <span>15 ~ 50 μm</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="font-medium">코팅 두께</span>
                          <span>0.5 ~ 2.0 μm</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="font-medium">인장 강도</span>
                          <span>≥ 120 MPa</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="font-medium">연신율</span>
                          <span>≥ 3.5%</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="font-medium">열팽창 계수</span>
                          <span>14 ~ 16 ppm/°C</span>
                        </li>
                      </ul>
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-navy-700 mb-4">전기적 특성</h3>
                      <ul className="space-y-3">
                        <li className="flex justify-between">
                          <span className="font-medium">절연 저항</span>
                          <span>≥ 10^9 Ω·cm</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="font-medium">절연 내압</span>
                          <span>≥ 100 V</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="font-medium">유전 상수</span>
                          <span>3.2 ~ 3.8</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="font-medium">유전 손실</span>
                          <span>≤ 0.02</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="font-medium">내열 온도</span>
                          <span>-65°C ~ 200°C</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </Card>
              </TabsContent>
              <TabsContent value="applications">
                <Card className="p-6">
                  <div className="grid md:grid-cols-2 gap-8">
                    <div>
                      <h3 className="text-xl font-bold text-navy-700 mb-4">주요 적용 분야</h3>
                      <ul className="space-y-3">
                        <li className="flex items-start">
                          <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                          <div>
                            <span className="font-medium">고집적 반도체 패키징</span>
                            <p className="text-sm text-gray-600 mt-1">
                              미세 피치 적용이 가능하여 고집적 반도체 패키징에 최적화
                            </p>
                          </div>
                        </li>
                        <li className="flex items-start">
                          <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                          <div>
                            <span className="font-medium">고주파 회로</span>
                            <p className="text-sm text-gray-600 mt-1">
                              우수한 절연 특성으로 고주파 회로에서의 신호 간섭 최소화
                            </p>
                          </div>
                        </li>
                        <li className="flex items-start">
                          <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                          <div>
                            <span className="font-medium">고온 환경 애플리케이션</span>
                            <p className="text-sm text-gray-600 mt-1">
                              높은 내열성으로 자동차, 항공우주 등 고온 환경에서 사용 가능
                            </p>
                          </div>
                        </li>
                      </ul>
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-navy-700 mb-4">산업별 활용</h3>
                      <ul className="space-y-3">
                        <li className="flex items-start">
                          <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                          <div>
                            <span className="font-medium">모바일 기기</span>
                            <p className="text-sm text-gray-600 mt-1">
                              스마트폰, 태블릿 등 소형 전자기기의 고밀도 패키징
                            </p>
                          </div>
                        </li>
                        <li className="flex items-start">
                          <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                          <div>
                            <span className="font-medium">자동차 전자장치</span>
                            <p className="text-sm text-gray-600 mt-1">고온 환경에서 작동하는 자동차 ECU 및 센서 모듈</p>
                          </div>
                        </li>
                        <li className="flex items-start">
                          <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                          <div>
                            <span className="font-medium">의료 기기</span>
                            <p className="text-sm text-gray-600 mt-1">
                              고신뢰성이 요구되는 의료용 임플란트 및 진단 장비
                            </p>
                          </div>
                        </li>
                      </ul>
                    </div>
                  </div>
                </Card>
              </TabsContent>
              <TabsContent value="comparison">
                <Card className="p-6">
                  <h3 className="text-xl font-bold text-navy-700 mb-4">경쟁 제품 비교</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr className="bg-navy-50">
                          <th className="border border-gray-200 p-3 text-left">특성</th>
                          <th className="border border-gray-200 p-3 text-left">JP SCIENCE Insulation Wire</th>
                          <th className="border border-gray-200 p-3 text-left">일반 Bonding Wire</th>
                          <th className="border border-gray-200 p-3 text-left">경쟁사 A 제품</th>
                          <th className="border border-gray-200 p-3 text-left">경쟁사 B 제품</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td className="border border-gray-200 p-3 font-medium">절연 저항</td>
                          <td className="border border-gray-200 p-3 text-navy-600">≥ 10^9 Ω·cm</td>
                          <td className="border border-gray-200 p-3">해당 없음</td>
                          <td className="border border-gray-200 p-3">10^7 Ω·cm</td>
                          <td className="border border-gray-200 p-3">10^8 Ω·cm</td>
                        </tr>
                        <tr>
                          <td className="border border-gray-200 p-3 font-medium">내열 온도</td>
                          <td className="border border-gray-200 p-3 text-navy-600">-65°C ~ 200°C</td>
                          <td className="border border-gray-200 p-3">-55°C ~ 150°C</td>
                          <td className="border border-gray-200 p-3">-60°C ~ 180°C</td>
                          <td className="border border-gray-200 p-3">-60°C ~ 170°C</td>
                        </tr>
                        <tr>
                          <td className="border border-gray-200 p-3 font-medium">최소 피치</td>
                          <td className="border border-gray-200 p-3 text-navy-600">35 μm</td>
                          <td className="border border-gray-200 p-3">50 μm</td>
                          <td className="border border-gray-200 p-3">40 μm</td>
                          <td className="border border-gray-200 p-3">45 μm</td>
                        </tr>
                        <tr>
                          <td className="border border-gray-200 p-3 font-medium">신뢰성 (HTSL)</td>
                          <td className="border border-gray-200 p-3 text-navy-600">1000시간 @ 175°C</td>
                          <td className="border border-gray-200 p-3">500시간 @ 150°C</td>
                          <td className="border border-gray-200 p-3">800시간 @ 170°C</td>
                          <td className="border border-gray-200 p-3">700시간 @ 165°C</td>
                        </tr>
                        <tr>
                          <td className="border border-gray-200 p-3 font-medium">코팅 균일성</td>
                          <td className="border border-gray-200 p-3 text-navy-600">± 0.1 μm</td>
                          <td className="border border-gray-200 p-3">해당 없음</td>
                          <td className="border border-gray-200 p-3">± 0.2 μm</td>
                          <td className="border border-gray-200 p-3">± 0.25 μm</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </section>

        {/* Product Gallery Section */}
        <section className="py-16">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">제품 갤러리</h2>
              <p className="mt-4 text-gray-600">다양한 제품 이미지</p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="relative h-48 rounded-lg overflow-hidden">
                <Image
                  src="/placeholder.svg?height=200&width=300"
                  alt="제품 이미지 1"
                  fill
                  className="object-cover hover:scale-105 transition-transform duration-300"
                />
              </div>
              <div className="relative h-48 rounded-lg overflow-hidden">
                <Image
                  src="/placeholder.svg?height=200&width=300"
                  alt="제품 이미지 2"
                  fill
                  className="object-cover hover:scale-105 transition-transform duration-300"
                />
              </div>
              <div className="relative h-48 rounded-lg overflow-hidden">
                <Image
                  src="/placeholder.svg?height=200&width=300"
                  alt="제품 이미지 3"
                  fill
                  className="object-cover hover:scale-105 transition-transform duration-300"
                />
              </div>
              <div className="relative h-48 rounded-lg overflow-hidden">
                <Image
                  src="/placeholder.svg?height=200&width=300"
                  alt="제품 이미지 4"
                  fill
                  className="object-cover hover:scale-105 transition-transform duration-300"
                />
              </div>
            </div>
          </div>
        </section>

        {/* Product Inquiry Section */}
        <section className="py-16 bg-gray-50">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">제품 문의</h2>
              <p className="mt-4 text-gray-600">제품에 대한 자세한 정보나 견적을 원하시면 문의해 주세요.</p>
            </div>

            <div className="flex justify-center">
              <Card className="p-6 max-w-md w-full">
                <div className="space-y-4 text-center">
                  <p className="text-gray-700">
                    JP SCIENCE의 제품에 관심을 가져주셔서 감사합니다. 제품 사양, 가격, 납기 등에 대한 자세한 정보를
                    원하시면 아래 버튼을 클릭하여 문의해 주세요.
                  </p>
                  <div className="pt-4">
                    <Link href="/contact">
                      <Button className="bg-navy-600 hover:bg-navy-700 text-white w-full">
                        제품 문의하기 <ArrowRight className="ml-2 h-4 w-4" />
                      </Button>
                    </Link>
                  </div>
                  <div className="pt-2">
                    <p className="text-sm text-gray-600">
                      또는 전화 (02-2220-0911) 또는 이메일 (info@jpscience.com)로 문의하실 수 있습니다.
                    </p>
                  </div>
                </div>
              </Card>
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
