import Image from "next/image"
import Link from "next/link"
import {
  Mail,
  MapPin,
  Phone,
  ChevronRight,
  ArrowRight,
  Beaker,
  Microscope,
  Database,
  FlaskRoundIcon as Flask,
  Award,
  FileText,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"

export default function Home() {
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
              alt="Laboratory equipment"
              fill
              className="object-cover brightness-50"
              priority
            />
          </div>
          <div className="container relative z-10 py-24 md:py-32 lg:py-40">
            <div className="max-w-3xl space-y-5 text-white">
              <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">
                끊임없는 선구적인 연구개발과 혁신
              </h1>
              <p className="text-lg md:text-xl">
                JP SCIENCE는 첨단 과학 기술을 통해 미래를 선도합니다. 우리의 혁신적인 연구와 개발은 산업의 새로운 표준을
                만들어갑니다.
              </p>
              <div className="pt-4">
                <Button className="bg-navy-600 hover:bg-navy-700 text-white">
                  서비스 문의하기 <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </section>

        {/* CEO Greeting Section */}
        <section id="about" className="py-16 bg-gray-50">
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
                <p className="font-semibold text-navy-700">홍진표 드림</p>
                <div className="pt-4">
                  <Link href="/about">
                    <Button variant="outline" className="text-navy-600 border-navy-600 hover:bg-navy-50">
                      회사 소개 더 보기 <ChevronRight className="ml-2 h-4 w-4" />
                    </Button>
                  </Link>
                </div>
              </div>
            </div>
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

        {/* Services Section */}
        <section id="services" className="py-16">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">서비스</h2>
              <p className="mt-4 text-gray-600">JP SCIENCE의 전문 서비스</p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              {/* Service 1 */}
              <Card className="p-6 hover:shadow-lg transition-shadow border-t-4 border-t-navy-600">
                <div className="w-12 h-12 rounded-full bg-navy-100 flex items-center justify-center mb-4">
                  <Beaker className="h-6 w-6 text-navy-600" />
                </div>
                <h3 className="text-xl font-bold text-navy-700">Thin Film Deposition</h3>
                <p className="mt-2 text-gray-700">
                  최첨단 장비를 활용한 고품질 박막 증착 서비스를 제공합니다. 반도체, 디스플레이, 광학 소자 등 다양한
                  분야에 적용 가능합니다.
                </p>
                <div className="mt-4">
                  <Link href="/services" className="text-navy-600 hover:text-navy-700 inline-flex items-center">
                    자세히 보기 <ChevronRight className="ml-1 h-4 w-4" />
                  </Link>
                </div>
              </Card>

              {/* Service 2 */}
              <Card className="p-6 hover:shadow-lg transition-shadow border-t-4 border-t-navy-600">
                <div className="w-12 h-12 rounded-full bg-navy-100 flex items-center justify-center mb-4">
                  <Microscope className="h-6 w-6 text-navy-600" />
                </div>
                <h3 className="text-xl font-bold text-navy-700">HR-XRD 측정 & 분석</h3>
                <p className="mt-2 text-gray-700">
                  고해상도 X선 회절 분석을 통해 재료의 결정 구조, 두께, 조성 등을 정밀하게 측정하고 분석합니다. 정확한
                  데이터를 바탕으로 최적의 솔루션을 제공합니다.
                </p>
                <div className="mt-4">
                  <Link href="/services" className="text-navy-600 hover:text-navy-700 inline-flex items-center">
                    자세히 보기 <ChevronRight className="ml-1 h-4 w-4" />
                  </Link>
                </div>
              </Card>

              {/* Service 3 */}
              <Card className="p-6 hover:shadow-lg transition-shadow border-t-4 border-t-navy-600">
                <div className="w-12 h-12 rounded-full bg-navy-100 flex items-center justify-center mb-4">
                  <Database className="h-6 w-6 text-navy-600" />
                </div>
                <h3 className="text-xl font-bold text-navy-700">Ion Milling</h3>
                <p className="mt-2 text-gray-700">
                  이온 밀링 기술을 활용한 정밀 가공 서비스를 제공합니다. 나노 수준의 정밀도로 다양한 재료의 표면 처리 및
                  미세 가공이 가능합니다.
                </p>
                <div className="mt-4">
                  <Link href="/services" className="text-navy-600 hover:text-navy-700 inline-flex items-center">
                    자세히 보기 <ChevronRight className="ml-1 h-4 w-4" />
                  </Link>
                </div>
              </Card>
            </div>

            <div className="mt-10 text-center">
              <Link href="/services">
                <Button className="bg-navy-600 hover:bg-navy-700 text-white">
                  모든 서비스 보기 <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </div>
          </div>
        </section>

        {/* Products Section */}
        <section id="products" className="py-16 bg-gray-50">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">제품</h2>
              <p className="mt-4 text-gray-600">JP SCIENCE의 혁신적인 제품</p>
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
                  <Link href="/products">
                    <Button className="bg-navy-600 hover:bg-navy-700 text-white">
                      제품 상세 정보 <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Technology Section */}
        <section className="py-16">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">기술력</h2>
              <p className="mt-4 text-gray-600">JP SCIENCE의 핵심 기술과 연구 성과</p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
              {/* Stat 1 */}
              <div className="p-6 bg-white rounded-lg shadow-sm">
                <div className="text-4xl font-bold text-navy-700">15+</div>
                <p className="mt-2 text-gray-600">특허 보유</p>
              </div>

              {/* Stat 2 */}
              <div className="p-6 bg-white rounded-lg shadow-sm">
                <div className="text-4xl font-bold text-navy-700">30+</div>
                <p className="mt-2 text-gray-600">연구 논문</p>
              </div>

              {/* Stat 3 */}
              <div className="p-6 bg-white rounded-lg shadow-sm">
                <div className="text-4xl font-bold text-navy-700">50+</div>
                <p className="mt-2 text-gray-600">기업 협력</p>
              </div>

              {/* Stat 4 */}
              <div className="p-6 bg-white rounded-lg shadow-sm">
                <div className="text-4xl font-bold text-navy-700">8+</div>
                <p className="mt-2 text-gray-600">연구 분야</p>
              </div>
            </div>

            <div className="mt-16 grid md:grid-cols-3 gap-8">
              {/* Tech Area 1 */}
              <Card className="p-6 hover:shadow-lg transition-shadow">
                <h3 className="text-xl font-bold text-navy-700">반도체 소재</h3>
                <p className="mt-2 text-gray-700">
                  차세대 반도체 소재 개발 및 특성 분석을 통해 고성능 반도체 제조 기술을 연구합니다.
                </p>
              </Card>

              {/* Tech Area 2 */}
              <Card className="p-6 hover:shadow-lg transition-shadow">
                <h3 className="text-xl font-bold text-navy-700">나노 기술</h3>
                <p className="mt-2 text-gray-700">
                  나노 스케일의 재료 및 구조 제어 기술을 개발하여 다양한 산업 분야에 적용합니다.
                </p>
              </Card>

              {/* Tech Area 3 */}
              <Card className="p-6 hover:shadow-lg transition-shadow">
                <h3 className="text-xl font-bold text-navy-700">측정 및 분석</h3>
                <p className="mt-2 text-gray-700">
                  첨단 측정 장비와 분석 기술을 활용하여 정밀한 데이터 수집 및 해석을 제공합니다.
                </p>
              </Card>
            </div>
          </div>
        </section>

        {/* Contact Section */}
        <section id="contact" className="py-16 bg-gray-50">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">문의하기</h2>
              <p className="mt-4 text-gray-600">JP SCIENCE에 궁금한 점이 있으시면 언제든지 문의해 주세요.</p>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              <div className="space-y-6">
                <div className="flex items-start">
                  <div className="w-12 h-12 rounded-full bg-navy-100 flex items-center justify-center mr-4">
                    <MapPin className="h-6 w-6 text-navy-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-navy-700">주소</h3>
                    <p className="mt-1 text-gray-600">서울특별시 성동구 왕십리로 222 한양대학교 자연과학관 428호</p>
                  </div>
                </div>

                <div className="flex items-start">
                  <div className="w-12 h-12 rounded-full bg-navy-100 flex items-center justify-center mr-4">
                    <Phone className="h-6 w-6 text-navy-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-navy-700">전화</h3>
                    <p className="mt-1 text-gray-600">02-2220-0911</p>
                  </div>
                </div>

                <div className="flex items-start">
                  <div className="w-12 h-12 rounded-full bg-navy-100 flex items-center justify-center mr-4">
                    <Mail className="h-6 w-6 text-navy-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-navy-700">이메일</h3>
                    <p className="mt-1 text-gray-600">info@jpscience.com</p>
                  </div>
                </div>

                <div className="pt-6">
                  <iframe
                    src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3162.6884077102397!2d127.04192491531878!3d37.55777797979957!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x357ca4947a71cf69%3A0x9e6aa3ada8573322!2z7ZWc7JaR64yA7ZWZ6rWQIOyekOuPmeqzte2Vmeq0gA!5e0!3m2!1sko!2skr!4v1650000000000!5m2!1sko!2skr"
                    width="100%"
                    height="250"
                    style={{ border: 0 }}
                    allowFullScreen
                    loading="lazy"
                    referrerPolicy="no-referrer-when-downgrade"
                    className="rounded-lg"
                  ></iframe>
                </div>
              </div>

              <div>
                <Card className="p-6">
                  <h3 className="text-xl font-bold text-navy-700 mb-4">문의 양식</h3>
                  <form className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <label htmlFor="name" className="text-sm font-medium">
                          이름
                        </label>
                        <Input id="name" placeholder="이름을 입력하세요" />
                      </div>
                      <div className="space-y-2">
                        <label htmlFor="email" className="text-sm font-medium">
                          이메일
                        </label>
                        <Input id="email" type="email" placeholder="이메일을 입력하세요" />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="subject" className="text-sm font-medium">
                        제목
                      </label>
                      <Input id="subject" placeholder="제목을 입력하세요" />
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="message" className="text-sm font-medium">
                        메시지
                      </label>
                      <Textarea id="message" placeholder="메시지를 입력하세요" rows={5} />
                    </div>
                    <Button className="w-full bg-navy-600 hover:bg-navy-700 text-white">보내기</Button>
                  </form>
                </Card>
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
                <Input placeholder="이메일을 입력하세요" className="bg-navy-700 border-navy-600 text-white" />
                <Button className="ml-2 bg-navy-600 hover:bg-navy-500">구독</Button>
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
