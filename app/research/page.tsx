import Image from "next/image"
import Link from "next/link"
import {
  Mail,
  MapPin,
  Phone,
  ChevronRight,
  Beaker,
  FlaskRoundIcon as Flask,
  Microscope,
  Award,
  FileText,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

export default function ResearchPage() {
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
            <Link
              href="/research"
              className="text-sm font-medium text-navy-600 border-b-2 border-navy-600 transition-colors"
            >
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
              alt="Research Laboratory"
              fill
              className="object-cover brightness-50"
              priority
            />
          </div>
          <div className="container relative z-10 py-24 md:py-32">
            <div className="max-w-3xl space-y-5 text-white">
              <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">연구개발</h1>
              <p className="text-lg md:text-xl">
                JP SCIENCE는 첨단 과학 기술을 통해 미래를 선도합니다. 우리의 혁신적인 연구와 개발은 산업의 새로운 표준을
                만들어갑니다.
              </p>
            </div>
          </div>
        </section>

        {/* Current Research Milestone Section */}
        <section className="py-16 bg-gray-50">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">현재 연구 프로젝트</h2>
              <p className="mt-4 text-gray-600">Insulating Bonding Wire 코팅 기술 개발 마일스톤</p>
            </div>

            <div className="grid md:grid-cols-2 gap-8 mb-12">
              <div className="space-y-4">
                <h3 className="text-xl font-bold text-navy-700">연구 개요</h3>
                <p className="text-gray-700">
                  JP SCIENCE는 반도체 패키징 공정에서 사용되는 고성능 Insulating Bonding Wire의 코팅 방법과 코팅 재료에
                  대한 연구를 진행하고 있습니다. 이 연구는 와이어의 절연 특성과 내구성을 향상시키고, 고집적 반도체
                  제품에 최적화된 솔루션을 제공하는 것을 목표로 합니다.
                </p>
                <div className="mt-4">
                  <h4 className="font-semibold text-navy-700">주요 연구 목표</h4>
                  <ul className="mt-2 space-y-2">
                    <li className="flex items-start">
                      <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                      <span>고효율 절연 코팅 재료 개발</span>
                    </li>
                    <li className="flex items-start">
                      <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                      <span>균일한 코팅 두께를 위한 공정 최적화</span>
                    </li>
                    <li className="flex items-start">
                      <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                      <span>고온 환경에서의 내구성 향상</span>
                    </li>
                    <li className="flex items-start">
                      <ChevronRight className="h-5 w-5 text-navy-600 mr-2 mt-0.5" />
                      <span>미세 피치 적용을 위한 코팅 기술 개발</span>
                    </li>
                  </ul>
                </div>
              </div>
              <div className="relative h-[300px] rounded-lg overflow-hidden">
                <Image
                  src="/placeholder.svg?height=300&width=500"
                  alt="Insulating Bonding Wire Research"
                  fill
                  className="object-cover"
                />
              </div>
            </div>

            <Card className="p-6">
              <h3 className="text-xl font-bold text-navy-700 mb-4">연구 마일스톤</h3>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[180px]">단계</TableHead>
                    <TableHead>연구 내용</TableHead>
                    <TableHead>기간</TableHead>
                    <TableHead className="text-right">진행 상황</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow>
                    <TableCell className="font-medium">1단계: 코팅 재료 선정</TableCell>
                    <TableCell>다양한 절연 재료의 특성 분석 및 최적 재료 선정</TableCell>
                    <TableCell>2024.07 ~ 2024.09</TableCell>
                    <TableCell className="text-right">완료</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">2단계: 코팅 방법 개발</TableCell>
                    <TableCell>균일한 코팅을 위한 공정 개발 및 최적화</TableCell>
                    <TableCell>2024.10 ~ 2025.01</TableCell>
                    <TableCell className="text-right">진행 중 (80%)</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">3단계: 특성 평가</TableCell>
                    <TableCell>코팅된 와이어의 전기적, 기계적, 열적 특성 평가</TableCell>
                    <TableCell>2025.02 ~ 2025.04</TableCell>
                    <TableCell className="text-right">예정</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">4단계: 신뢰성 테스트</TableCell>
                    <TableCell>다양한 환경 조건에서의 장기 신뢰성 테스트</TableCell>
                    <TableCell>2025.05 ~ 2025.07</TableCell>
                    <TableCell className="text-right">예정</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">5단계: 양산 기술 개발</TableCell>
                    <TableCell>대량 생산을 위한 공정 최적화 및 품질 관리 시스템 구축</TableCell>
                    <TableCell>2025.08 ~ 2025.12</TableCell>
                    <TableCell className="text-right">예정</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </Card>
          </div>
        </section>

        {/* R&D Timeline Section */}
        <section id="research" className="py-16">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">연구개발 타임라인</h2>
              <p className="mt-4 text-gray-600">JP SCIENCE의 혁신적인 연구 여정</p>
            </div>

            <div className="relative">
              {/* Vertical line */}
              <div className="absolute left-1/2 transform -translate-x-1/2 h-full w-0.5 bg-navy-200"></div>

              {/* Timeline items */}
              <div className="space-y-12">
                {/* 2023 */}
                <div className="relative">
                  <div className="flex items-center justify-center">
                    <div className="absolute z-10 w-10 h-10 rounded-full bg-navy-600 flex items-center justify-center left-1/2 transform -translate-x-1/2">
                      <Beaker className="h-5 w-5 text-white" />
                    </div>
                  </div>
                  <div className="ml-auto mr-auto md:ml-auto md:mr-0 md:pr-8 w-full md:w-1/2 mt-8 relative">
                    <Card className="p-6 hover:shadow-lg transition-shadow border-l-4 border-l-navy-600">
                      <div className="absolute -left-3 top-4 md:hidden w-6 h-0.5 bg-navy-200"></div>
                      <h3 className="text-xl font-bold text-navy-700">2023</h3>
                      <div className="mt-2 space-y-3">
                        <p className="text-gray-700">
                          결함 제어 기반 단일 셀 선택 메모리 이중 기능 selector-only memory 소자 원천 기술 연구 (23.02 ~
                          26.01)
                        </p>
                        <p className="text-gray-700">
                          고속/저전력/고신뢰성의 2단자 field-free SOT-MRAM 원천 기술 개빌 및 IP 확보 (23.04 ~ 27.12)
                        </p>
                      </div>
                    </Card>
                  </div>
                </div>

                {/* 2021 */}
                <div className="relative">
                  <div className="flex items-center justify-center">
                    <div className="absolute z-10 w-10 h-10 rounded-full bg-navy-600 flex items-center justify-center left-1/2 transform -translate-x-1/2">
                      <Flask className="h-5 w-5 text-white" />
                    </div>
                  </div>
                  <div className="ml-auto mr-auto md:mr-auto md:ml-0 md:pl-8 w-full md:w-1/2 mt-8 relative">
                    <Card className="p-6 hover:shadow-lg transition-shadow border-l-4 border-l-navy-600">
                      <div className="absolute -right-3 top-4 md:hidden w-6 h-0.5 bg-navy-200"></div>
                      <h3 className="text-xl font-bold text-navy-700">2021</h3>
                      <div className="mt-2 space-y-3">
                        <p className="text-gray-700">
                          자기 스커미온을 이용한 멀티레벨 뇌기능 모사 소자 (21.04 ~ 23.12)
                        </p>
                        <p className="text-gray-700">
                          인체 삽입형 의료기기 자가 충전을 위한 고신축성 에너지 하베스팅 양극 소재 및 물성 원천기술 연구
                          (21.03 ~ 24.02)
                        </p>
                      </div>
                    </Card>
                  </div>
                </div>

                {/* 2019 */}
                <div className="relative">
                  <div className="flex items-center justify-center">
                    <div className="absolute z-10 w-10 h-10 rounded-full bg-navy-600 flex items-center justify-center left-1/2 transform -translate-x-1/2">
                      <Microscope className="h-5 w-5 text-white" />
                    </div>
                  </div>
                  <div className="ml-auto mr-auto md:ml-auto md:mr-0 md:pr-8 w-full md:w-1/2 mt-8 relative">
                    <Card className="p-6 hover:shadow-lg transition-shadow border-l-4 border-l-navy-600">
                      <div className="absolute -left-3 top-4 md:hidden w-6 h-0.5 bg-navy-200"></div>
                      <h3 className="text-xl font-bold text-navy-700">2019</h3>
                      <div className="mt-2 space-y-3">
                        <p className="text-gray-700">
                          대면적/집적화 가능한 complementary SOT-MTJ 구조기반 극 저 전력용 단위 셀 reconfigurable 스핀
                          로직 소자 원천기술 (19.06 ~ 21.12)
                        </p>
                      </div>
                    </Card>
                  </div>
                </div>

                {/* 2017 */}
                <div className="relative">
                  <div className="flex items-center justify-center">
                    <div className="absolute z-10 w-10 h-10 rounded-full bg-navy-600 flex items-center justify-center left-1/2 transform -translate-x-1/2">
                      <Award className="h-5 w-5 text-white" />
                    </div>
                  </div>
                  <div className="ml-auto mr-auto md:mr-auto md:ml-0 md:pl-8 w-full md:w-1/2 mt-8 relative">
                    <Card className="p-6 hover:shadow-lg transition-shadow border-l-4 border-l-navy-600">
                      <div className="absolute -right-3 top-4 md:hidden w-6 h-0.5 bg-navy-200"></div>
                      <h3 className="text-xl font-bold text-navy-700">2017</h3>
                      <div className="mt-2 space-y-3">
                        <p className="text-gray-700">
                          3차원 적층 cross-bar 수직형 스핀토크 자기저항 메모리 집적화 및 응용 기술 연구 (17.03 ~ 20.02)
                        </p>
                      </div>
                    </Card>
                  </div>
                </div>

                {/* 2016 */}
                <div className="relative">
                  <div className="flex items-center justify-center">
                    <div className="absolute z-10 w-10 h-10 rounded-full bg-navy-600 flex items-center justify-center left-1/2 transform -translate-x-1/2">
                      <FileText className="h-5 w-5 text-white" />
                    </div>
                  </div>
                  <div className="ml-auto mr-auto md:ml-auto md:mr-0 md:pr-8 w-full md:w-1/2 mt-8 relative">
                    <Card className="p-6 hover:shadow-lg transition-shadow border-l-4 border-l-navy-600">
                      <div className="absolute -left-3 top-4 md:hidden w-6 h-0.5 bg-navy-200"></div>
                      <h3 className="text-xl font-bold text-navy-700">2016</h3>
                      <div className="mt-2 space-y-3">
                        <p className="text-gray-700">
                          인체의 움직임으로 동작하는 1차원 Fiber 기반 에너지 하베스팅 의복화 연구 (16.12 ~ 17.11)
                        </p>
                        <p className="text-gray-700">Forming-free a-C:Ox 기반 ReRAM 원천 기술 개발 (16.10 ~ 21.09)</p>
                        <p className="text-gray-700">
                          나노전자소자 기술을 응용한 신경세포 모방 뉴런소자 및 시스템 원천기술 개발 (16.08 ~ 21.07)
                        </p>
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
