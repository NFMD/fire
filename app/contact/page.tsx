import Image from "next/image"
import Link from "next/link"
import { Mail, MapPin, Phone, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"

export default function ContactPage() {
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
            <Link
              href="/contact"
              className="text-sm font-medium text-navy-600 border-b-2 border-navy-600 transition-colors"
            >
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
              alt="Contact Us"
              fill
              className="object-cover brightness-50"
              priority
            />
          </div>
          <div className="container relative z-10 py-24 md:py-32">
            <div className="max-w-3xl space-y-5 text-white">
              <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">문의하기</h1>
              <p className="text-lg md:text-xl">
                JP SCIENCE에 궁금한 점이 있으시면 언제든지 문의해 주세요. 최대한 빠르게 답변 드리겠습니다.
              </p>
            </div>
          </div>
        </section>

        {/* Contact Information Section */}
        <section className="py-16 bg-gray-50">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">연락처 정보</h2>
              <p className="mt-4 text-gray-600">다양한 방법으로 JP SCIENCE에 연락하실 수 있습니다.</p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              <Card className="p-6 text-center hover:shadow-lg transition-shadow">
                <div className="w-16 h-16 rounded-full bg-navy-100 flex items-center justify-center mx-auto mb-4">
                  <MapPin className="h-8 w-8 text-navy-600" />
                </div>
                <h3 className="text-xl font-bold text-navy-700 mb-2">주소</h3>
                <p className="text-gray-700">서울특별시 성동구 왕십리로 222 한양대학교 자연과학관 428호</p>
              </Card>

              <Card className="p-6 text-center hover:shadow-lg transition-shadow">
                <div className="w-16 h-16 rounded-full bg-navy-100 flex items-center justify-center mx-auto mb-4">
                  <Phone className="h-8 w-8 text-navy-600" />
                </div>
                <h3 className="text-xl font-bold text-navy-700 mb-2">전화</h3>
                <p className="text-gray-700">02-2220-0911</p>
                <p className="text-sm text-gray-600 mt-2">평일 09:00 - 18:00</p>
              </Card>

              <Card className="p-6 text-center hover:shadow-lg transition-shadow">
                <div className="w-16 h-16 rounded-full bg-navy-100 flex items-center justify-center mx-auto mb-4">
                  <Mail className="h-8 w-8 text-navy-600" />
                </div>
                <h3 className="text-xl font-bold text-navy-700 mb-2">이메일</h3>
                <p className="text-gray-700">info@jpscience.com</p>
                <p className="text-sm text-gray-600 mt-2">24시간 이내 답변 드립니다</p>
              </Card>
            </div>
          </div>
        </section>

        {/* Contact Form Section */}
        <section className="py-16">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">문의 양식</h2>
              <p className="mt-4 text-gray-600">아래 양식을 작성하여 문의해 주세요.</p>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
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
                      <label htmlFor="phone" className="text-sm font-medium">
                        연락처
                      </label>
                      <Input id="phone" placeholder="연락처를 입력하세요" />
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="subject" className="text-sm font-medium">
                        제목
                      </label>
                      <Input id="subject" placeholder="제목을 입력하세요" />
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="inquiry-type" className="text-sm font-medium">
                        문의 유형
                      </label>
                      <select
                        id="inquiry-type"
                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        <option value="">문의 유형을 선택하세요</option>
                        <option value="product">제품 문의</option>
                        <option value="service">서비스 문의</option>
                        <option value="research">연구 협력 문의</option>
                        <option value="career">채용 문의</option>
                        <option value="other">기타</option>
                      </select>
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="message" className="text-sm font-medium">
                        메시지
                      </label>
                      <Textarea id="message" placeholder="문의 내용을 입력하세요" rows={5} />
                    </div>
                    <Button className="w-full bg-navy-600 hover:bg-navy-700 text-white">
                      보내기 <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  </form>
                </Card>
              </div>

              <div className="space-y-8">
                <div className="relative h-[300px] rounded-lg overflow-hidden">
                  <Image src="/placeholder.svg?height=300&width=500" alt="Map Location" fill className="object-cover" />
                </div>

                <Card className="p-6">
                  <h3 className="text-xl font-bold text-navy-700 mb-4">자주 묻는 질문</h3>
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-semibold text-navy-700">문의 후 답변은 얼마나 걸리나요?</h4>
                      <p className="text-gray-700 mt-1">
                        일반적으로 24시간 이내에 답변 드리고 있습니다. 복잡한 문의의 경우 추가 시간이 소요될 수
                        있습니다.
                      </p>
                    </div>
                    <div>
                      <h4 className="font-semibold text-navy-700">제품 견적은 어떻게 요청하나요?</h4>
                      <p className="text-gray-700 mt-1">
                        문의 양식에서 '제품 문의'를 선택하시고 필요한 제품과 수량을 기재해 주시면 견적을 보내드립니다.
                      </p>
                    </div>
                    <div>
                      <h4 className="font-semibold text-navy-700">연구 협력은 어떻게 진행되나요?</h4>
                      <p className="text-gray-700 mt-1">
                        연구 협력 문의를 주시면 담당자가 연락드려 구체적인 협력 방안에 대해 논의하게 됩니다.
                      </p>
                    </div>
                  </div>
                </Card>
              </div>
            </div>
          </div>
        </section>

        {/* Map Section */}
        <section className="py-16 bg-gray-50">
          <div className="container">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-navy-700">찾아오시는 길</h2>
              <p className="mt-4 text-gray-600">JP SCIENCE 오시는 방법</p>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <div className="h-[400px] rounded-lg overflow-hidden">
                  <iframe
                    src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3162.6884077102397!2d127.04192491531878!3d37.55777797979957!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x357ca4947a71cf69%3A0x9e6aa3ada8573322!2z7ZWc7JaR64yA7ZWZ6rWQIOyekOuPmeqzte2Vmeq0gA!5e0!3m2!1sko!2skr!4v1650000000000!5m2!1sko!2skr"
                    width="100%"
                    height="400"
                    style={{ border: 0 }}
                    allowFullScreen
                    loading="lazy"
                    referrerPolicy="no-referrer-when-downgrade"
                    className="rounded-lg"
                  ></iframe>
                </div>
              </div>
              <div>
                <Card className="p-6 h-full">
                  <h3 className="text-xl font-bold text-navy-700 mb-4">교통 안내</h3>
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-semibold text-navy-700">지하철</h4>
                      <p className="text-gray-700 mt-1">2호선 한양대역 2번 출구에서 도보 5분</p>
                    </div>
                    <div>
                      <h4 className="font-semibold text-navy-700">버스</h4>
                      <p className="text-gray-700 mt-1">지선버스: 121, 302, 2012, 2014, 2016, 2222</p>
                      <p className="text-gray-700 mt-1">간선버스: 110, 302, N62</p>
                    </div>
                    <div>
                      <h4 className="font-semibold text-navy-700">자가용</h4>
                      <p className="text-gray-700 mt-1">한양대학교 정문으로 진입 후 안내에 따라 자연과학관으로 이동</p>
                      <p className="text-gray-700 mt-1">주차: 한양대학교 지하주차장 이용 (유료)</p>
                    </div>
                    <div>
                      <h4 className="font-semibold text-navy-700">찾아오시는 길 상세 안내</h4>
                      <p className="text-gray-700 mt-1">
                        한양대학교 자연과학관 4층 428호에 위치해 있습니다. 건물 입구에서 엘리베이터를 이용하여 4층으로
                        오신 후 안내표시를 따라오시면 됩니다.
                      </p>
                    </div>
                  </div>
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
