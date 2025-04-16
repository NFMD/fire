import Image from "next/image"
import Link from "next/link"
import {
  Mail,
  MapPin,
  Phone,
  ChevronRight,
  ArrowRight,
  FileText,
  Layers,
  Atom,
  Search,
  CheckCircle2,
  Cog,
  ClipboardList,
  BarChart,
  FileCheck,
  HeartHandshake,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function ServicesPage() {
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
            <Link
              href="/services"
              className="text-sm font-medium text-navy-600 border-b-2 border-navy-600 transition-colors"
            >
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
          <div className="container relative z-10 py-24 md:py-32">
            <div className="max-w-3xl space-y-5 text-white">
              <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">전문적인 기술 서비스</h1>
              <p className="text-lg md:text-xl">
                JP SCIENCE는 첨단 과학 기술을 바탕으로 고객의 요구에 맞는 최적의 솔루션을 제공합니다. 우리의 전문성과
                경험을 통해 고객의 연구와 개발을 성공으로 이끌어 드립니다.
              </p>
              <div className="pt-4">
                <Button className="bg-navy-600 hover:bg-navy-700 text-white">
                  서비스 문의하기 <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </section>

        {/* Main Services Section */}
        <section className="py-16">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">주요 서비스</h2>
              <p className="mt-4 text-gray-600">JP SCIENCE의 핵심 기술 서비스를 소개합니다</p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              {/* Service 1 */}
              <Card className="overflow-hidden hover:shadow-lg transition-shadow">
                <div className="relative h-48">
                  <Image
                    src="/placeholder.svg?height=300&width=500"
                    alt="박막 증착 기술"
                    fill
                    className="object-cover"
                  />
                </div>
                <div className="p-6">
                  <div className="w-12 h-12 rounded-full bg-navy-100 flex items-center justify-center mb-4">
                    <Layers className="h-6 w-6 text-navy-600" />
                  </div>
                  <h3 className="text-xl font-bold text-navy-700">박막 증착 기술 서비스</h3>
                  <p className="mt-2 text-gray-700">
                    최첨단 장비를 활용한 고품질 박막 증착 서비스를 제공합니다. 반도체, 디스플레이, 광학 소자 등 다양한
                    분야에 적용 가능한 맞춤형 솔루션을 제공합니다.
                  </p>
                  <ul className="mt-4 space-y-2">
                    <li className="flex items-start">
                      <CheckCircle2 className="h-5 w-5 text-navy-600 mr-2 mt-0.5 shrink-0" />
                      <span>Thin Film Deposition</span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle2 className="h-5 w-5 text-navy-600 mr-2 mt-0.5 shrink-0" />
                      <span>다층 박막 구조 설계</span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle2 className="h-5 w-5 text-navy-600 mr-2 mt-0.5 shrink-0" />
                      <span>맞춤형 공정 개발</span>
                    </li>
                  </ul>
                  <div className="mt-6">
                    <Button className="bg-navy-600 hover:bg-navy-700 text-white w-full">
                      자세히 보기 <ChevronRight className="ml-2 h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </Card>

              {/* Service 2 */}
              <Card className="overflow-hidden hover:shadow-lg transition-shadow">
                <div className="relative h-48">
                  <Image src="/placeholder.svg?height=300&width=500" alt="신소재 개발" fill className="object-cover" />
                </div>
                <div className="p-6">
                  <div className="w-12 h-12 rounded-full bg-navy-100 flex items-center justify-center mb-4">
                    <Atom className="h-6 w-6 text-navy-600" />
                  </div>
                  <h3 className="text-xl font-bold text-navy-700">신소재 개발 서비스</h3>
                  <p className="mt-2 text-gray-700">
                    나노 기술을 활용한 신소재 개발 및 특성 분석 서비스를 제공합니다. 고객의 요구에 맞는 맞춤형 소재
                    개발과 최적화를 지원합니다.
                  </p>
                  <ul className="mt-4 space-y-2">
                    <li className="flex items-start">
                      <CheckCircle2 className="h-5 w-5 text-navy-600 mr-2 mt-0.5 shrink-0" />
                      <span>나노 소재 합성</span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle2 className="h-5 w-5 text-navy-600 mr-2 mt-0.5 shrink-0" />
                      <span>기능성 소재 개발</span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle2 className="h-5 w-5 text-navy-600 mr-2 mt-0.5 shrink-0" />
                      <span>소재 특성 최적화</span>
                    </li>
                  </ul>
                  <div className="mt-6">
                    <Button className="bg-navy-600 hover:bg-navy-700 text-white w-full">
                      자세히 보기 <ChevronRight className="ml-2 h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </Card>

              {/* Service 3 */}
              <Card className="overflow-hidden hover:shadow-lg transition-shadow">
                <div className="relative h-48">
                  <Image src="/placeholder.svg?height=300&width=500" alt="분석 서비스" fill className="object-cover" />
                </div>
                <div className="p-6">
                  <div className="w-12 h-12 rounded-full bg-navy-100 flex items-center justify-center mb-4">
                    <Search className="h-6 w-6 text-navy-600" />
                  </div>
                  <h3 className="text-xl font-bold text-navy-700">분석 서비스</h3>
                  <p className="mt-2 text-gray-700">
                    첨단 분석 장비를 활용한 정밀 분석 서비스를 제공합니다. 재료의 구조, 조성, 특성 등을 정확하게
                    분석하여 연구 및 개발에 필요한 데이터를 제공합니다.
                  </p>
                  <ul className="mt-4 space-y-2">
                    <li className="flex items-start">
                      <CheckCircle2 className="h-5 w-5 text-navy-600 mr-2 mt-0.5 shrink-0" />
                      <span>HR-XRD 측정 및 분석</span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle2 className="h-5 w-5 text-navy-600 mr-2 mt-0.5 shrink-0" />
                      <span>표면 분석</span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle2 className="h-5 w-5 text-navy-600 mr-2 mt-0.5 shrink-0" />
                      <span>전기적/광학적 특성 분석</span>
                    </li>
                  </ul>
                  <div className="mt-6">
                    <Button className="bg-navy-600 hover:bg-navy-700 text-white w-full">
                      자세히 보기 <ChevronRight className="ml-2 h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </Card>
            </div>
          </div>
        </section>

        {/* Service Process Section */}
        <section className="py-16 bg-gray-50">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">서비스 프로세스</h2>
              <p className="mt-4 text-gray-600">JP SCIENCE의 체계적인 서비스 진행 과정</p>
            </div>

            <div className="relative">
              {/* Horizontal line for desktop */}
              <div className="hidden md:block absolute top-1/2 left-0 right-0 h-0.5 bg-navy-200 transform -translate-y-1/2"></div>

              <div className="grid md:grid-cols-5 gap-8">
                {/* Step 1 */}
                <div className="relative flex flex-col items-center">
                  <div className="z-10 w-16 h-16 rounded-full bg-navy-600 flex items-center justify-center mb-4">
                    <ClipboardList className="h-8 w-8 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-navy-700 text-center">상담 및 요구사항 분석</h3>
                  <p className="mt-2 text-gray-600 text-center">
                    고객의 요구사항을 정확히 파악하고 최적의 서비스 방향을 설정합니다.
                  </p>
                </div>

                {/* Step 2 */}
                <div className="relative flex flex-col items-center">
                  <div className="z-10 w-16 h-16 rounded-full bg-navy-600 flex items-center justify-center mb-4">
                    <FileText className="h-8 w-8 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-navy-700 text-center">서비스 제안 및 계약</h3>
                  <p className="mt-2 text-gray-600 text-center">
                    요구사항에 맞는 서비스 계획을 제안하고 세부 사항을 협의합니다.
                  </p>
                </div>

                {/* Step 3 */}
                <div className="relative flex flex-col items-center">
                  <div className="z-10 w-16 h-16 rounded-full bg-navy-600 flex items-center justify-center mb-4">
                    <Cog className="h-8 w-8 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-navy-700 text-center">연구/개발 수행</h3>
                  <p className="mt-2 text-gray-600 text-center">
                    전문 인력과 첨단 장비를 활용하여 체계적으로 연구 및 개발을 진행합니다.
                  </p>
                </div>

                {/* Step 4 */}
                <div className="relative flex flex-col items-center">
                  <div className="z-10 w-16 h-16 rounded-full bg-navy-600 flex items-center justify-center mb-4">
                    <BarChart className="h-8 w-8 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-navy-700 text-center">결과 분석 및 보고</h3>
                  <p className="mt-2 text-gray-600 text-center">
                    연구 결과를 정밀하게 분석하고 상세한 보고서를 제공합니다.
                  </p>
                </div>

                {/* Step 5 */}
                <div className="relative flex flex-col items-center">
                  <div className="z-10 w-16 h-16 rounded-full bg-navy-600 flex items-center justify-center mb-4">
                    <HeartHandshake className="h-8 w-8 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-navy-700 text-center">사후 관리</h3>
                  <p className="mt-2 text-gray-600 text-center">
                    서비스 완료 후에도 지속적인 기술 지원과 사후 관리를 제공합니다.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Equipment Section */}
        <section className="py-16">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">주요 장비 소개</h2>
              <p className="mt-4 text-gray-600">JP SCIENCE의 첨단 연구 및 분석 장비</p>
            </div>

            <Tabs defaultValue="deposition" className="w-full">
              <TabsList className="grid w-full grid-cols-3 mb-8">
                <TabsTrigger value="deposition">박막 증착 장비</TabsTrigger>
                <TabsTrigger value="analysis">분석 장비</TabsTrigger>
                <TabsTrigger value="processing">가공 장비</TabsTrigger>
              </TabsList>
              <TabsContent value="deposition">
                <div className="grid md:grid-cols-3 gap-6">
                  {/* Equipment 1 */}
                  <Card className="overflow-hidden">
                    <div className="relative h-48">
                      <Image
                        src="/placeholder.svg?height=300&width=500"
                        alt="Sputtering System"
                        fill
                        className="object-cover"
                      />
                    </div>
                    <div className="p-4">
                      <h3 className="text-lg font-bold text-navy-700">Sputtering System</h3>
                      <p className="text-sm text-gray-600 mt-1">Model: JP-SPT-2000</p>
                      <div className="mt-2">
                        <p className="text-sm text-gray-700">
                          <span className="font-semibold">주요 스펙:</span> 8인치 타겟, 4개 소스, RF/DC 전원
                        </p>
                        <p className="text-sm text-gray-700 mt-1">
                          <span className="font-semibold">활용 분야:</span> 반도체, 디스플레이, 광학 소자
                        </p>
                      </div>
                    </div>
                  </Card>

                  {/* Equipment 2 */}
                  <Card className="overflow-hidden">
                    <div className="relative h-48">
                      <Image
                        src="/placeholder.svg?height=300&width=500"
                        alt="E-beam Evaporator"
                        fill
                        className="object-cover"
                      />
                    </div>
                    <div className="p-4">
                      <h3 className="text-lg font-bold text-navy-700">E-beam Evaporator</h3>
                      <p className="text-sm text-gray-600 mt-1">Model: JP-EBE-1500</p>
                      <div className="mt-2">
                        <p className="text-sm text-gray-700">
                          <span className="font-semibold">주요 스펙:</span> 6-pocket 소스, 10kW 전원, 고진공 시스템
                        </p>
                        <p className="text-sm text-gray-700 mt-1">
                          <span className="font-semibold">활용 분야:</span> 금속 박막, 유전체 코팅
                        </p>
                      </div>
                    </div>
                  </Card>

                  {/* Equipment 3 */}
                  <Card className="overflow-hidden">
                    <div className="relative h-48">
                      <Image
                        src="/placeholder.svg?height=300&width=500"
                        alt="MOCVD System"
                        fill
                        className="object-cover"
                      />
                    </div>
                    <div className="p-4">
                      <h3 className="text-lg font-bold text-navy-700">MOCVD System</h3>
                      <p className="text-sm text-gray-600 mt-1">Model: JP-MOCVD-3000</p>
                      <div className="mt-2">
                        <p className="text-sm text-gray-700">
                          <span className="font-semibold">주요 스펙:</span> 다중 가스 라인, 정밀 온도 제어, 자동화
                          시스템
                        </p>
                        <p className="text-sm text-gray-700 mt-1">
                          <span className="font-semibold">활용 분야:</span> 화합물 반도체, LED, 태양전지
                        </p>
                      </div>
                    </div>
                  </Card>
                </div>
              </TabsContent>
              <TabsContent value="analysis">
                <div className="grid md:grid-cols-3 gap-6">
                  {/* Analysis Equipment 1 */}
                  <Card className="overflow-hidden">
                    <div className="relative h-48">
                      <Image src="/placeholder.svg?height=300&width=500" alt="HR-XRD" fill className="object-cover" />
                    </div>
                    <div className="p-4">
                      <h3 className="text-lg font-bold text-navy-700">HR-XRD</h3>
                      <p className="text-sm text-gray-600 mt-1">Model: JP-XRD-5000</p>
                      <div className="mt-2">
                        <p className="text-sm text-gray-700">
                          <span className="font-semibold">주요 스펙:</span> 고해상도, 다양한 측정 모드, 자동화 분석
                        </p>
                        <p className="text-sm text-gray-700 mt-1">
                          <span className="font-semibold">활용 분야:</span> 결정 구조 분석, 박막 두께 측정
                        </p>
                      </div>
                    </div>
                  </Card>

                  {/* Analysis Equipment 2 */}
                  <Card className="overflow-hidden">
                    <div className="relative h-48">
                      <Image src="/placeholder.svg?height=300&width=500" alt="SEM" fill className="object-cover" />
                    </div>
                    <div className="p-4">
                      <h3 className="text-lg font-bold text-navy-700">SEM</h3>
                      <p className="text-sm text-gray-600 mt-1">Model: JP-SEM-4000</p>
                      <div className="mt-2">
                        <p className="text-sm text-gray-700">
                          <span className="font-semibold">주요 스펙:</span> 고배율, EDS 분석 기능, 고해상도 이미징
                        </p>
                        <p className="text-sm text-gray-700 mt-1">
                          <span className="font-semibold">활용 분야:</span> 표면 형상 분석, 원소 분석
                        </p>
                      </div>
                    </div>
                  </Card>

                  {/* Analysis Equipment 3 */}
                  <Card className="overflow-hidden">
                    <div className="relative h-48">
                      <Image src="/placeholder.svg?height=300&width=500" alt="AFM" fill className="object-cover" />
                    </div>
                    <div className="p-4">
                      <h3 className="text-lg font-bold text-navy-700">AFM</h3>
                      <p className="text-sm text-gray-600 mt-1">Model: JP-AFM-2000</p>
                      <div className="mt-2">
                        <p className="text-sm text-gray-700">
                          <span className="font-semibold">주요 스펙:</span> 나노미터 해상도, 다양한 측정 모드, 3D 이미징
                        </p>
                        <p className="text-sm text-gray-700 mt-1">
                          <span className="font-semibold">활용 분야:</span> 표면 거칠기 분석, 나노 구조 측정
                        </p>
                      </div>
                    </div>
                  </Card>
                </div>
              </TabsContent>
              <TabsContent value="processing">
                <div className="grid md:grid-cols-3 gap-6">
                  {/* Processing Equipment 1 */}
                  <Card className="overflow-hidden">
                    <div className="relative h-48">
                      <Image
                        src="/placeholder.svg?height=300&width=500"
                        alt="Ion Milling System"
                        fill
                        className="object-cover"
                      />
                    </div>
                    <div className="p-4">
                      <h3 className="text-lg font-bold text-navy-700">Ion Milling System</h3>
                      <p className="text-sm text-gray-600 mt-1">Model: JP-IMS-1000</p>
                      <div className="mt-2">
                        <p className="text-sm text-gray-700">
                          <span className="font-semibold">주요 스펙:</span> 정밀 각도 제어, 다중 이온 소스, 자동화
                          시스템
                        </p>
                        <p className="text-sm text-gray-700 mt-1">
                          <span className="font-semibold">활용 분야:</span> 미세 가공, 표면 처리
                        </p>
                      </div>
                    </div>
                  </Card>

                  {/* Processing Equipment 2 */}
                  <Card className="overflow-hidden">
                    <div className="relative h-48">
                      <Image
                        src="/placeholder.svg?height=300&width=500"
                        alt="Plasma Etcher"
                        fill
                        className="object-cover"
                      />
                    </div>
                    <div className="p-4">
                      <h3 className="text-lg font-bold text-navy-700">Plasma Etcher</h3>
                      <p className="text-sm text-gray-600 mt-1">Model: JP-PE-3000</p>
                      <div className="mt-2">
                        <p className="text-sm text-gray-700">
                          <span className="font-semibold">주요 스펙:</span> 다양한 가스 지원, 고밀도 플라즈마, 정밀 제어
                        </p>
                        <p className="text-sm text-gray-700 mt-1">
                          <span className="font-semibold">활용 분야:</span> 반도체 공정, 미세 패턴 형성
                        </p>
                      </div>
                    </div>
                  </Card>

                  {/* Processing Equipment 3 */}
                  <Card className="overflow-hidden">
                    <div className="relative h-48">
                      <Image
                        src="/placeholder.svg?height=300&width=500"
                        alt="Laser Processing System"
                        fill
                        className="object-cover"
                      />
                    </div>
                    <div className="p-4">
                      <h3 className="text-lg font-bold text-navy-700">Laser Processing System</h3>
                      <p className="text-sm text-gray-600 mt-1">Model: JP-LPS-2000</p>
                      <div className="mt-2">
                        <p className="text-sm text-gray-700">
                          <span className="font-semibold">주요 스펙:</span> 고출력 레이저, 정밀 위치 제어, 다양한 파장
                        </p>
                        <p className="text-sm text-gray-700 mt-1">
                          <span className="font-semibold">활용 분야:</span> 정밀 절단, 패터닝, 표면 개질
                        </p>
                      </div>
                    </div>
                  </Card>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </section>

        {/* Service Application Guide */}
        <section className="py-16 bg-gray-50">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">서비스 신청 가이드</h2>
              <p className="mt-4 text-gray-600">JP SCIENCE 서비스 이용 방법</p>
            </div>

            <div className="grid md:grid-cols-2 gap-12">
              <div>
                <h3 className="text-xl font-bold text-navy-700 mb-4">신청 방법</h3>
                <ol className="space-y-4">
                  <li className="flex">
                    <span className="w-8 h-8 rounded-full bg-navy-600 text-white flex items-center justify-center mr-3 shrink-0">
                      1
                    </span>
                    <div>
                      <p className="font-semibold">문의하기</p>
                      <p className="text-gray-600 mt-1">
                        웹사이트, 이메일, 전화를 통해 원하는 서비스에 대해 문의합니다.
                      </p>
                    </div>
                  </li>
                  <li className="flex">
                    <span className="w-8 h-8 rounded-full bg-navy-600 text-white flex items-center justify-center mr-3 shrink-0">
                      2
                    </span>
                    <div>
                      <p className="font-semibold">상담 및 요구사항 전달</p>
                      <p className="text-gray-600 mt-1">담당자와의 상담을 통해 구체적인 요구사항을 전달합니다.</p>
                    </div>
                  </li>
                  <li className="flex">
                    <span className="w-8 h-8 rounded-full bg-navy-600 text-white flex items-center justify-center mr-3 shrink-0">
                      3
                    </span>
                    <div>
                      <p className="font-semibold">견적 및 일정 확인</p>
                      <p className="text-gray-600 mt-1">요구사항에 맞는 서비스 견적과 일정을 확인합니다.</p>
                    </div>
                  </li>
                  <li className="flex">
                    <span className="w-8 h-8 rounded-full bg-navy-600 text-white flex items-center justify-center mr-3 shrink-0">
                      4
                    </span>
                    <div>
                      <p className="font-semibold">계약 체결</p>
                      <p className="text-gray-600 mt-1">서비스 내용, 비용, 일정 등에 대한 계약을 체결합니다.</p>
                    </div>
                  </li>
                  <li className="flex">
                    <span className="w-8 h-8 rounded-full bg-navy-600 text-white flex items-center justify-center mr-3 shrink-0">
                      5
                    </span>
                    <div>
                      <p className="font-semibold">서비스 진행 및 완료</p>
                      <p className="text-gray-600 mt-1">계약에 따라 서비스를 진행하고 결과물을 전달받습니다.</p>
                    </div>
                  </li>
                </ol>
              </div>

              <div>
                <h3 className="text-xl font-bold text-navy-700 mb-4">필요 서류 및 정보</h3>
                <Card className="p-6">
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <FileCheck className="h-5 w-5 text-navy-600 mr-2 mt-0.5 shrink-0" />
                      <span>서비스 신청서 (웹사이트에서 다운로드 가능)</span>
                    </li>
                    <li className="flex items-start">
                      <FileCheck className="h-5 w-5 text-navy-600 mr-2 mt-0.5 shrink-0" />
                      <span>사업자등록증 사본 (기업 고객의 경우)</span>
                    </li>
                    <li className="flex items-start">
                      <FileCheck className="h-5 w-5 text-navy-600 mr-2 mt-0.5 shrink-0" />
                      <span>연구 목적 및 요구사항 상세 명세서</span>
                    </li>
                    <li className="flex items-start">
                      <FileCheck className="h-5 w-5 text-navy-600 mr-2 mt-0.5 shrink-0" />
                      <span>기존 연구 자료 (해당되는 경우)</span>
                    </li>
                  </ul>

                  <div className="mt-6 pt-6 border-t">
                    <h4 className="font-semibold text-navy-700 mb-3">견적 산정 기준</h4>
                    <ul className="space-y-2 text-gray-700">
                      <li>• 서비스 종류 및 복잡도</li>
                      <li>• 필요 장비 및 재료</li>
                      <li>• 소요 시간 및 인력</li>
                      <li>• 분석 및 보고서 요구 수준</li>
                    </ul>
                  </div>

                  <div className="mt-6 pt-6 border-t">
                    <h4 className="font-semibold text-navy-700 mb-3">일반적인 소요 기간</h4>
                    <ul className="space-y-2 text-gray-700">
                      <li>• 기본 분석 서비스: 1-2주</li>
                      <li>• 박막 증착 서비스: 2-4주</li>
                      <li>• 신소재 개발: 1-3개월</li>
                      <li>• 맞춤형 연구 프로젝트: 3-6개월</li>
                    </ul>
                  </div>
                </Card>
              </div>
            </div>
          </div>
        </section>

        {/* Customer Cases */}
        <section className="py-16">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">고객 사례</h2>
              <p className="mt-4 text-gray-600">JP SCIENCE와 함께한 성공 사례</p>
            </div>

            <div className="mb-12">
              <h3 className="text-xl font-bold text-navy-700 mb-6">주요 협력 기업</h3>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="bg-gray-100 h-20 rounded-md flex items-center justify-center">
                    <Image
                      src={`/placeholder.svg?height=60&width=120&text=Partner ${i}`}
                      alt={`Partner ${i}`}
                      width={120}
                      height={60}
                    />
                  </div>
                ))}
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              {/* Case Study 1 */}
              <Card className="overflow-hidden">
                <div className="p-6">
                  <h3 className="text-lg font-bold text-navy-700">반도체 소재 개발 프로젝트</h3>
                  <p className="text-sm text-gray-600 mt-1">A 전자 기업</p>
                  <p className="mt-4 text-gray-700">
                    차세대 반도체 소자용 신소재 개발 프로젝트를 성공적으로 수행하여 고객사의 제품 성능을 20%
                    향상시켰습니다.
                  </p>
                  <div className="mt-4 flex items-center">
                    <div className="flex-1">
                      <div className="text-sm font-semibold text-navy-700">프로젝트 기간</div>
                      <div className="text-sm text-gray-600">6개월</div>
                    </div>
                    <div className="flex-1">
                      <div className="text-sm font-semibold text-navy-700">서비스 유형</div>
                      <div className="text-sm text-gray-600">신소재 개발</div>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Case Study 2 */}
              <Card className="overflow-hidden">
                <div className="p-6">
                  <h3 className="text-lg font-bold text-navy-700">광학 필름 코팅 최적화</h3>
                  <p className="text-sm text-gray-600 mt-1">B 디스플레이 기업</p>
                  <p className="mt-4 text-gray-700">
                    특수 광학 필름 코팅 공정을 최적화하여 제품의 광학적 특성을 개선하고 생산 효율을 30% 향상시켰습니다.
                  </p>
                  <div className="mt-4 flex items-center">
                    <div className="flex-1">
                      <div className="text-sm font-semibold text-navy-700">프로젝트 기간</div>
                      <div className="text-sm text-gray-600">3개월</div>
                    </div>
                    <div className="flex-1">
                      <div className="text-sm font-semibold text-navy-700">서비스 유형</div>
                      <div className="text-sm text-gray-600">박막 증착 최적화</div>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Case Study 3 */}
              <Card className="overflow-hidden">
                <div className="p-6">
                  <h3 className="text-lg font-bold text-navy-700">에너지 저장 소재 분석</h3>
                  <p className="text-sm text-gray-600 mt-1">C 에너지 기업</p>
                  <p className="mt-4 text-gray-700">
                    차세대 배터리용 소재의 구조 및 특성 분석을 통해 성능 저하 원인을 규명하고 개선 방안을 제시했습니다.
                  </p>
                  <div className="mt-4 flex items-center">
                    <div className="flex-1">
                      <div className="text-sm font-semibold text-navy-700">프로젝트 기간</div>
                      <div className="text-sm text-gray-600">2개월</div>
                    </div>
                    <div className="flex-1">
                      <div className="text-sm font-semibold text-navy-700">서비스 유형</div>
                      <div className="text-sm text-gray-600">분석 서비스</div>
                    </div>
                  </div>
                </div>
              </Card>
            </div>

            <div className="mt-12">
              <h3 className="text-xl font-bold text-navy-700 mb-6">고객 후기</h3>
              <div className="grid md:grid-cols-2 gap-8">
                {/* Testimonial 1 */}
                <Card className="p-6">
                  <div className="flex items-center mb-4">
                    <div className="w-12 h-12 rounded-full bg-gray-200 mr-4"></div>
                    <div>
                      <p className="font-semibold">김OO 연구소장</p>
                      <p className="text-sm text-gray-600">A 전자 기업</p>
                    </div>
                  </div>
                  <p className="text-gray-700 italic">
                    "JP SCIENCE의 전문적인 기술력과 체계적인 연구 방법론 덕분에 우리 회사의 신제품 개발 기간을 크게
                    단축할 수 있었습니다. 특히 연구 과정에서의 소통과 피드백이 매우 만족스러웠습니다."
                  </p>
                </Card>

                {/* Testimonial 2 */}
                <Card className="p-6">
                  <div className="flex items-center mb-4">
                    <div className="w-12 h-12 rounded-full bg-gray-200 mr-4"></div>
                    <div>
                      <p className="font-semibold">이OO 책임연구원</p>
                      <p className="text-sm text-gray-600">C 에너지 기업</p>
                    </div>
                  </div>
                  <p className="text-gray-700 italic">
                    "복잡한 소재 분석 문제를 정확하게 해결해 주셔서 감사합니다. JP SCIENCE의 첨단 장비와 전문 인력의
                    분석 결과는 우리 연구팀에게 큰 도움이 되었습니다. 앞으로도 지속적인 협력을 기대합니다."
                  </p>
                </Card>
              </div>
            </div>
          </div>
        </section>

        {/* Contact Section */}
        <section id="contact" className="py-16 bg-gray-50">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">문의하기</h2>
              <p className="mt-4 text-gray-600">
                JP SCIENCE 서비스에 대해 궁금한 점이 있으시면 언제든지 문의해 주세요.
              </p>
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
                    <p className="mt-1 text-gray-600">service@jpscience.com</p>
                  </div>
                </div>

                <div className="pt-6">
                  <iframe
                    src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3165.346195630915!2d127.0283357!3d37.4969958!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x0%3A0x0!2zMzfCsDI5JzQ5LjEiTiAxMjfCsDAyJzA2LjAiRQ!5e0!3m2!1sen!2skr!4v1650000000000!5m2!1sen!2skr"
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
                  <h3 className="text-xl font-bold text-navy-700 mb-4">서비스 문의 양식</h3>
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
                      <label htmlFor="phone" className="text-sm font-medium">
                        연락처
                      </label>
                      <Input id="phone" placeholder="연락처를 입력하세요" />
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="service" className="text-sm font-medium">
                        관심 서비스
                      </label>
                      <select
                        id="service"
                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        <option value="">서비스를 선택하세요</option>
                        <option value="deposition">박막 증착 서비스</option>
                        <option value="material">신소재 개발 서비스</option>
                        <option value="analysis">분석 서비스</option>
                        <option value="other">기타</option>
                      </select>
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="message" className="text-sm font-medium">
                        문의 내용
                      </label>
                      <Textarea id="message" placeholder="문의 내용을 입력하세요" rows={5} />
                    </div>
                    <Button className="w-full bg-navy-600 hover:bg-navy-700 text-white">문의하기</Button>
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
