import Image from "next/image"
import Link from "next/link"
import { Mail, MapPin, Phone, Calendar, User, Eye, ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

// 더미 공지사항 데이터
const noticeData = [
  {
    id: 1,
    title: "JP SCIENCE 홈페이지 리뉴얼 안내",
    date: "2024-07-15",
    author: "관리자",
    content: `안녕하세요, JP SCIENCE입니다.

저희 홈페이지가 새롭게 리뉴얼되었습니다. 더욱 편리한 서비스를 제공하기 위해 최선을 다하겠습니다.

<주요 개선 사항>
1. 반응형 웹 디자인 적용으로 모바일 환경에서도 편리하게 이용 가능
2. 공지사항 게시판 신설로 회사 소식 및 중요 공지 확인 가능
3. 제품 및 서비스 정보 상세화로 필요한 정보를 쉽게 확인 가능
4. 문의하기 기능 개선으로 더욱 빠른 응대 가능

앞으로도 JP SCIENCE는 고객 여러분께 더 나은 서비스를 제공하기 위해 노력하겠습니다.
많은 관심과 이용 부탁드립니다.

감사합니다.`,
    views: 245,
    attachments: [{ name: "홈페이지_이용_가이드.pdf", size: "2.3MB" }],
  },
  {
    id: 2,
    title: "2024년 하반기 연구개발 프로젝트 공고",
    date: "2024-07-10",
    author: "연구개발팀",
    content: `JP SCIENCE에서 2024년 하반기 연구개발 프로젝트를 공고합니다.

<프로젝트 개요>
- 프로젝트명: 고효율 절연 코팅 재료 개발
- 연구 기간: 2024년 8월 ~ 2025년 1월 (6개월)
- 연구 내용: 반도체 패키징용 고효율 절연 코팅 재료 개발 및 특성 평가
- 지원 자격: 관련 분야 연구 경험이 있는 기업 및 연구기관

<지원 방법>
- 신청서 제출 기한: 2024년 7월 31일까지
- 제출 서류: 연구 계획서, 회사 소개서, 연구 실적 자료
- 제출 방법: 이메일 (research@jpscience.com)

<선정 절차>
- 1차 서류 심사: 2024년 8월 1일 ~ 8월 5일
- 2차 발표 심사: 2024년 8월 10일
- 최종 선정 발표: 2024년 8월 15일

관심 있는 기업 및 연구기관의 많은 참여 바랍니다.
문의사항은 연구개발팀(02-2220-0922)으로 연락 주시기 바랍니다.`,
    views: 189,
    attachments: [
      { name: "연구개발_프로젝트_신청서.docx", size: "1.5MB" },
      { name: "연구계획서_양식.docx", size: "0.8MB" },
    ],
  },
  {
    id: 3,
    title: "신규 장비 도입 안내",
    date: "2024-06-28",
    author: "장비운영팀",
    content: `JP SCIENCE에서 최신 HR-XRD 장비를 도입하였습니다.

<도입 장비 정보>
- 장비명: High Resolution X-Ray Diffractometer (HR-XRD)
- 모델명: JP-XRD-5000
- 주요 특징:
  1. 고해상도 X선 회절 분석 가능
  2. 다양한 측정 모드 지원
  3. 자동화된 분석 시스템 탑재
  4. 미세 구조 및 결정 특성 정밀 분석

<활용 분야>
- 박막의 결정 구조 분석
- 다층 박막의 두께 및 계면 특성 분석
- 나노 소재의 결정성 및 배향성 분석
- 반도체 소자의 결함 및 스트레스 분석

이번 장비 도입을 통해 더욱 정밀한 분석 서비스를 제공할 예정입니다.
장비 이용을 원하시는 고객께서는 장비운영팀(02-2220-0933)으로 문의 바랍니다.

감사합니다.`,
    views: 156,
    attachments: [
      { name: "HR-XRD_장비사양서.pdf", size: "3.2MB" },
      { name: "분석서비스_이용안내.pdf", size: "1.1MB" },
    ],
  },
  {
    id: 4,
    title: "하계 휴무 안내",
    date: "2024-06-20",
    author: "관리자",
    content: `JP SCIENCE 하계 휴무를 안내드립니다.

<휴무 기간>
- 2024년 8월 1일(목) ~ 8월 5일(월), 총 5일간

<휴무 기간 중 문의 안내>
- 긴급 문의: 비상연락망(010-1234-5678)
- 일반 문의: 이메일(info@jpscience.com)로 접수 (휴무 종료 후 순차적으로 답변 드립니다)

<주의사항>
- 휴무 기간 중에는 장비 이용 및 분석 서비스가 제공되지 않습니다.
- 휴무 전후 업무가 집중될 수 있으니, 필요한 서비스는 미리 신청해 주시기 바랍니다.
- 8월 6일(화)부터 정상 업무가 재개됩니다.

고객 여러분의 양해 부탁드립니다.
감사합니다.`,
    views: 132,
    attachments: [],
  },
  {
    id: 5,
    title: "Insulation Bonding Wire 신제품 출시",
    date: "2024-06-15",
    author: "제품개발팀",
    content: `JP SCIENCE에서 개발한 신제품 Insulation Bonding Wire가 출시되었습니다.

<제품 특징>
- 제품명: JP-IBW-2000
- 주요 특징:
  1. 기존 제품 대비 내열성 30% 향상 (최대 200°C 지원)
  2. 절연 저항 10^9 Ω·cm 이상 구현
  3. 균일한 코팅 두께(±0.1μm) 제공
  4. 미세 피치(35μm) 적용 가능

<적용 분야>
- 고집적 반도체 패키징
- 고주파 회로
- 고온 환경 애플리케이션
- 자동차 전자장치
- 의료 기기

<제품 출시 일정>
- 샘플 제공: 2024년 7월 1일부터
- 정식 판매: 2024년 8월 1일부터

제품에 관한 자세한 정보나 샘플 요청은 제품개발팀(product@jpscience.com)으로 문의 바랍니다.`,
    views: 201,
    attachments: [
      { name: "Insulation_Bonding_Wire_사양서.pdf", size: "2.7MB" },
      { name: "제품_적용사례.pdf", size: "4.5MB" },
    ],
  },
  {
    id: 6,
    title: "2024년 상반기 연구 성과 보고",
    date: "2024-06-05",
    author: "연구개발팀",
    content: `JP SCIENCE 2024년 상반기 연구 성과를 보고합니다.

<특허 출원>
1. "고효율 절연 코팅 방법 및 그 장치" (출원번호: 10-2024-0012345)
2. "나노 구조 제어를 통한 열전도도 향상 방법" (출원번호: 10-2024-0023456)
3. "반도체 패키징용 다층 절연 구조체" (출원번호: 10-2024-0034567)

<논문 게재>
1. "Effect of Coating Thickness on Insulation Properties of Bonding Wires" (Advanced Materials, IF: 27.398)
2. "Novel Approach for Uniform Coating of Nano-scale Insulation Layer" (ACS Nano, IF: 15.881)
3. "Thermal Stability Enhancement of Insulation Materials for Semiconductor Packaging" (Journal of Materials Chemistry, IF: 13.025)
4. "Characterization of Nano-structured Insulation Coating for High Temperature Applications" (Applied Physics Letters, IF: 3.791)
5. "Mechanical Properties of Multi-layered Insulation Structures for Electronic Devices" (Materials Science and Engineering, IF: 5.234)

<연구 진행 현황>
- 고신축성 에너지 하베스팅 소재 연구: 70% 완료
- 2단자 field-free SOT-MRAM 원천 기술 개발: 45% 완료
- 멀티레벨 뇌기능 모사 소자 연구: 85% 완료
- 신규 절연 코팅 재료 개발: 60% 완료

JP SCIENCE는 앞으로도 지속적인 연구개발을 통해 혁신적인 기술을 선보이며 글로벌 시장에서 경쟁력을 갖춘 기업으로 성장해 나갈 것입니다.

연구 성과에 관한 자세한 정보는 연구개발팀(research@jpscience.com)으로 문의 바랍니다.`,
    views: 178,
    attachments: [{ name: "2024_상반기_연구성과_요약.pdf", size: "5.2MB" }],
  },
  {
    id: 7,
    title: "산학협력 파트너십 체결 안내",
    date: "2024-05-20",
    author: "대외협력팀",
    content: `JP SCIENCE에서 한양대학교 공과대학과 산학협력 파트너십을 체결하였습니다.

<협력 내용>
- 공동 연구 프로젝트 수행
- 인재 양성 및 교육 프로그램 운영
- 연구 시설 및 장비 공동 활용
- 기술 세미나 및 워크샵 공동 개최

<기대 효과>
- 산학 간 기술 교류 활성화
- 실무 중심의 인재 양성
- 연구 개발 역량 강화
- 신기술 개발 및 사업화 촉진

이번 파트너십을 통해 JP SCIENCE는 학계의 최신 연구 동향을 파악하고, 한양대학교는 산업체의 실무 경험을 공유받는 상호 발전적인 관계를 구축할 예정입니다.

관련 문의사항은 대외협력팀(partnership@jpscience.com)으로 연락 주시기 바랍니다.`,
    views: 145,
    attachments: [{ name: "산학협력_MOU_보도자료.pdf", size: "1.8MB" }],
  },
  {
    id: 8,
    title: "채용 공고: 연구원 모집",
    date: "2024-05-10",
    author: "인사팀",
    content: `JP SCIENCE에서 신소재 개발 분야 연구원을 모집합니다.

<모집 분야 및 인원>
- 신소재 개발 연구원: 2명
- 분석 연구원: 1명

<자격 요건>
- 신소재 개발 연구원: 재료공학, 화학공학, 물리학 관련 석사 이상 학위 소지자
- 분석 연구원: 분석화학, 기기분석 관련 학사 이상 학위 소지자
- 공통: 관련 분야 연구 경험자 우대

<근무 조건>
- 근무지: 서울특별시 성동구 왕십리로 222 한양대학교 자연과학관 428호
- 급여: 내규에 따름 (경력에 따라 협의)
- 복리후생: 4대 보험, 퇴직금, 연구 성과급, 학회 참가 지원 등

<전형 절차>
- 서류 접수: 2024년 5월 10일 ~ 5월 31일
- 1차 서류 전형: 2024년 6월 1일 ~ 6월 5일
- 2차 면접 전형: 2024년 6월 10일 ~ 6월 15일
- 최종 합격 발표: 2024년 6월 20일
- 입사 예정일: 2024년 7월 1일

<제출 서류>
- 이력서 및 자기소개서
- 최종 학위 논문 요약본
- 연구 실적 목록
- 추천서(선택)

지원서 접수 및 문의: recruit@jpscience.com`,
    views: 223,
    attachments: [
      { name: "채용공고_상세요강.pdf", size: "1.2MB" },
      { name: "입사지원서_양식.docx", size: "0.5MB" },
    ],
  },
  {
    id: 9,
    title: "고객 만족도 조사 실시 안내",
    date: "2024-04-25",
    author: "고객지원팀",
    content: `JP SCIENCE 서비스 품질 향상을 위한 고객 만족도 조사를 실시합니다.

<조사 개요>
- 조사 기간: 2024년 5월 1일 ~ 5월 15일
- 조사 대상: JP SCIENCE 서비스 이용 고객
- 조사 방법: 온라인 설문조사
- 소요 시간: 약 10분

<조사 내용>
- 서비스 품질 만족도
- 제품 성능 및 품질 만족도
- 고객 응대 및 지원 만족도
- 개선 요청 사항

<참여 혜택>
- 설문 참여자 전원에게 모바일 커피 쿠폰 제공
- 추첨을 통해 10명에게 상품권(5만원) 증정

고객 여러분의 소중한 의견은 JP SCIENCE의 서비스 품질 향상을 위한 중요한 자료로 활용될 예정입니다.
많은 참여 부탁드립니다.

설문 참여 링크는 5월 1일 이메일로 발송될 예정입니다.
문의사항은 고객지원팀(support@jpscience.com)으로 연락 주시기 바랍니다.`,
    views: 98,
    attachments: [],
  },
  {
    id: 10,
    title: "2024년 기술 세미나 개최 안내",
    date: "2024-04-15",
    author: "교육팀",
    content: `JP SCIENCE에서 '첨단 소재 기술의 현재와 미래'를 주제로 기술 세미나를 개최합니다.

<세미나 개요>
- 일시: 2024년 5월 15일(수) 13:00 ~ 17:00
- 장소: 한양대학교 HIT 빌딩 6층 대회의실
- 주제: 첨단 소재 기술의 현재와 미래
- 참가비: 무료 (사전 등록 필수)

<프로그램>
- 13:00 ~ 13:30: 등록 및 개회사
- 13:30 ~ 14:20: [기조강연] "미래 산업을 이끌 첨단 소재 기술 동향" (홍진표 대표, JP SCIENCE)
- 14:30 ~ 15:20: "반도체 패키징용 절연 소재의 최신 연구 동향" (김OO 교수, 한양대학교)
- 15:30 ~ 16:20: "산업용 코팅 기술의 발전과 응용" (이OO 연구소장, A전자)
- 16:30 ~ 17:00: 패널 토론 및 Q&A

<참가 신청>
- 신청 기간: 2024년 4월 15일 ~ 5월 10일
- 신청 방법: 이메일(seminar@jpscience.com)로 신청서 제출
- 문의: 교육팀 02-2220-0944

관심 있는 분들의 많은 참여 바랍니다.`,
    views: 167,
    attachments: [
      { name: "세미나_안내문.pdf", size: "1.7MB" },
      { name: "참가신청서.docx", size: "0.4MB" },
      { name: "오시는_길.pdf", size: "0.9MB" },
    ],
  },
]

export default function NoticeDetailPage({ params }: { params: { id: string } }) {
  const noticeId = Number.parseInt(params.id)
  const notice = noticeData.find((item) => item.id === noticeId)

  if (!notice) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center">
        <h1 className="text-2xl font-bold text-navy-700">존재하지 않는 공지사항입니다.</h1>
        <Link href="/notice" className="mt-4 text-navy-600 hover:underline">
          공지사항 목록으로 돌아가기
        </Link>
      </div>
    )
  }

  // 이전 글, 다음 글 찾기
  const currentIndex = noticeData.findIndex((item) => item.id === noticeId)
  const prevNotice = currentIndex < noticeData.length - 1 ? noticeData[currentIndex + 1] : null
  const nextNotice = currentIndex > 0 ? noticeData[currentIndex - 1] : null

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

      <main className="flex-1 py-16">
        <div className="container">
          <div className="mb-6">
            <Link href="/notice" className="inline-flex items-center text-navy-600 hover:text-navy-700">
              <ArrowLeft className="mr-2 h-4 w-4" />
              공지사항 목록으로 돌아가기
            </Link>
          </div>

          <Card className="overflow-hidden">
            <div className="border-b p-6">
              <h1 className="text-2xl font-bold text-navy-700">{notice.title}</h1>
              <div className="mt-4 flex flex-wrap items-center gap-4 text-sm text-gray-600">
                <div className="flex items-center">
                  <User className="mr-2 h-4 w-4 text-gray-500" />
                  <span>{notice.author}</span>
                </div>
                <div className="flex items-center">
                  <Calendar className="mr-2 h-4 w-4 text-gray-500" />
                  <span>{notice.date}</span>
                </div>
                <div className="flex items-center">
                  <Eye className="mr-2 h-4 w-4 text-gray-500" />
                  <span>조회 {notice.views}</span>
                </div>
              </div>
            </div>
            <div className="p-6">
              <div className="prose max-w-none">
                {notice.content.split("\n\n").map((paragraph, index) => (
                  <p key={index} className="mb-4 whitespace-pre-line">
                    {paragraph}
                  </p>
                ))}
              </div>

              {notice.attachments && notice.attachments.length > 0 && (
                <div className="mt-8 border-t pt-6">
                  <h3 className="mb-4 font-semibold text-navy-700">첨부파일</h3>
                  <ul className="space-y-2">
                    {notice.attachments.map((file, index) => (
                      <li key={index} className="flex items-center">
                        <Button variant="outline" size="sm" className="inline-flex items-center">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className="mr-2 h-4 w-4"
                          >
                            <path d="M14 3v4a1 1 0 0 0 1 1h4" />
                            <path d="M17 21H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7l5 5v11a2 2 0 0 1-2 2z" />
                          </svg>
                          {file.name}
                          <span className="ml-2 text-xs text-gray-500">({file.size})</span>
                        </Button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </Card>

          <div className="mt-8">
            <Card>
              <div className="divide-y">
                {nextNotice && (
                  <div className="flex items-center justify-between p-4">
                    <div className="flex items-center">
                      <span className="mr-4 font-medium text-navy-700">다음 글</span>
                      <Link
                        href={`/notice/${nextNotice.id}`}
                        className="text-gray-700 hover:text-navy-600 hover:underline"
                      >
                        {nextNotice.title}
                      </Link>
                    </div>
                    <span className="text-sm text-gray-500">{nextNotice.date}</span>
                  </div>
                )}
                {prevNotice && (
                  <div className="flex items-center justify-between p-4">
                    <div className="flex items-center">
                      <span className="mr-4 font-medium text-navy-700">이전 글</span>
                      <Link
                        href={`/notice/${prevNotice.id}`}
                        className="text-gray-700 hover:text-navy-600 hover:underline"
                      >
                        {prevNotice.title}
                      </Link>
                    </div>
                    <span className="text-sm text-gray-500">{prevNotice.date}</span>
                  </div>
                )}
              </div>
            </Card>
          </div>

          <div className="mt-8 flex justify-center">
            <Link href="/notice">
              <Button className="bg-navy-600 hover:bg-navy-700 text-white">목록으로</Button>
            </Link>
          </div>
        </div>
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
